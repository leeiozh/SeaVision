import copy
from time import sleep, time
from queue import Queue, Empty, Full
from threading import Thread, Event, Lock
from src.io.structs import ProcessResult, Output
from src.runtime.logger import setup_logger

log = setup_logger("manager")

# Algorithm-state codes emitted in the UDP packet (byte 42, see udp_protocol.docx).
_STATE_WAIT  = 0   # waiting for data
_STATE_ACCUM = 1   # accumulating the initial buffer
_STATE_READY = 2   # operational / result ready
_STATE_RESET = 3   # reset / restart
_STATE_ERROR = 4   # error

_STATUS_INTERVAL = 1.0   # min seconds between throttled status heartbeats


def _drain(q: Queue):
    while not q.empty():
        try:
            q.get_nowait()
        except Empty:
            break


class Manager:
    """
    Three-thread pipeline: Input → Process → Output.

    Fault model:
      - Input timeout / EOF  → silent wait; after silence_threshold seconds marks reset_pending.
      - Data resumes         → resets processor state before feeding new data.
      - Processing exception → restarts processor thread immediately.
      - Processor hang       → watchdog (10 × rotation_period) restarts processor thread.
      - Output sink error    → logged per-sink, pipeline continues.
    """

    def __init__(self, config, processor_factory, inp_source, out_sinks: list):
        self.inp = inp_source
        self.out = out_sinks
        self.cfg = config

        self.processor_factory = processor_factory
        self.processor = None  # created in _start_processor

        queue_size = config.pipeline.queue_size
        self.in_queue: Queue = Queue(maxsize=queue_size)
        self.out_queue: Queue = Queue(maxsize=queue_size)

        self.stop_ev = Event()
        self.proc_stop_ev = Event()
        self.proc_lock = Lock()

        # RPM may be None (estimated live) — use 25 only for these coarse timeouts.
        rot_period = 60.0 / (config.const.RPM or 25)
        self._silence_threshold = config.const.N_SHOTS * rot_period
        self._watchdog_timeout = 10.0 * rot_period

        self._last_recv_time: float = 0.0
        self._reset_pending: bool = False
        self._proc_active_time: float = time()
        self._last_status_t: float = 0.0
        self._last_output = None   # last good Output — resent in heartbeats so the
                                   # receiver keeps its picture (only state/progress change)

        self.t_inp: Thread = None
        self.t_proc: Thread = None
        self.t_out: Thread = None

    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        log.info(
            f"Starting manager  silence={self._silence_threshold:.0f}s  "
            f"watchdog={self._watchdog_timeout:.0f}s"
        )
        self._start_processor()
        self.t_inp = Thread(target=self._input_loop, name="Input", daemon=True)
        self.t_out = Thread(target=self._output_loop, name="Output", daemon=True)
        self.t_inp.start()
        self.t_out.start()

    def stop(self):
        self._shutdown()

    # ── processor lifecycle ───────────────────────────────────────────────────

    def _start_processor(self):
        with self.proc_lock:
            # New Event each restart: old thread keeps its own (set) event and
            # exits cleanly; new thread gets a fresh (clear) event — no race.
            self.proc_stop_ev = Event()
            self.processor = self.processor_factory()
            self._proc_active_time = time()
            _stop = self.proc_stop_ev          # capture ref for closure
            self.t_proc = Thread(
                target=self._process_loop, args=(_stop,),
                name="Process", daemon=True)
            self.t_proc.start()

    def _restart_processor(self, reason: str):
        with self.proc_lock:
            if self.proc_stop_ev.is_set():
                return  # restart already in progress
            self.proc_stop_ev.set()

        print()  # end any in-progress \r progress bar before restart message
        log.warning(f"Restarting processor: {reason}")
        self._emit_status(_STATE_RESET, 0, force=True)
        _drain(self.in_queue)
        sleep(0.1)
        self._start_processor()

    # ── status heartbeats ─────────────────────────────────────────────────────

    def _emit_status(self, algo_state: int, progress: float, *, force: bool = False,
                     pulse: int = 0, step: float = 0.0):
        """Enqueue a status heartbeat so UDP sinks keep sending during waiting,
        accumulating, reset and error phases.

        To avoid blanking the receiver, the heartbeat carries the *last good
        result* (spectra and wave parameters) and only overrides the algo_state
        and progress bytes.  Before the first result exists there is nothing to
        show, so a zeroed status packet is sent instead.

        Throttled to one packet per _STATUS_INTERVAL unless force=True (used for
        one-shot RESET/ERROR transitions).  CSV sinks skip it (is_status=True).
        """
        now = time()
        if not force and (now - self._last_status_t) < _STATUS_INTERVAL:
            return
        self._last_status_t = now

        progress = max(0, min(100, int(progress)))
        last = self._last_output
        if last is not None:
            # Shallow copy: spectra/Wave refs are shared (sinks only read them,
            # _phys_clip is idempotent); the two overridden ints are per-copy.
            out = copy.copy(last)
            out.algo_state = algo_state
            out.progress = progress
        else:
            rps = float(getattr(self.processor, "rpm", None) or self.cfg.const.RPM or 25)
            out = Output.status(
                algo_state=algo_state, progress=progress,
                n_freq=self.cfg.const.N_FREQ, n_dirs=self.cfg.const.N_DIRS,
                n_freq_2d=self.cfg.const.N_FREQ_2D,
                pulse=pulse, step=step, rps=rps,
            )
        try:
            self.out_queue.put_nowait(ProcessResult(output=out, port=None, navi=None, is_status=True))
        except Full:
            pass

    # ── input thread ──────────────────────────────────────────────────────────

    def _input_loop(self):
        log.info("Input thread started")
        while not self.stop_ev.is_set():
            try:
                back = self.inp.get_bck()
            except TimeoutError:
                self._on_no_data()
                continue
            except Exception:
                log.exception("Input error")
                sleep(1.0)
                continue

            if back.step == 0.0:
                # EOF from file source — no new data available yet
                self._on_no_data()
                sleep(1.0)
                continue

            navi = self.inp.get_navi()

            # Coming back from a silence period → reset stale processor state
            if self._reset_pending:
                log.info("Data resumed after silence — resetting processor state")
                self._restart_processor("resumed after silence")
                self._reset_pending = False

            self._last_recv_time = time()
            self._check_processor_health()

            try:
                self.in_queue.put_nowait((back, navi))
            except Full:
                log.warning("Input queue full — frame dropped")

        log.warning("Input thread stopped")

    def _on_no_data(self):
        # Heartbeat so the external module shows "waiting for data" even before
        # the first frame ever arrives or during a silence period.
        self._emit_status(_STATE_WAIT, 0)
        if self._last_recv_time == 0.0:
            return  # no data has ever arrived, nothing to reset
        if not self._reset_pending and time() - self._last_recv_time > self._silence_threshold:
            log.warning(
                f"No data for >{self._silence_threshold:.0f}s — "
                "processor state will reset on next data"
            )
            self._reset_pending = True

    def _check_processor_health(self):
        # Unexpected thread death
        if (self.t_proc is not None
                and not self.t_proc.is_alive()
                and not self.proc_stop_ev.is_set()):
            self._restart_processor("processor thread died unexpectedly")
            return
        # Hung processor (one update taking too long)
        if time() - self._proc_active_time > self._watchdog_timeout:
            log.error("Processor watchdog timeout")
            self._restart_processor("watchdog timeout")

    # ── process thread ────────────────────────────────────────────────────────

    def _process_loop(self, stop_ev: Event):
        log.info("Process thread started")

        _n_shots   = self.cfg.const.N_SHOTS
        _out_times = max(1, int(self.cfg.output.get('out_times', 32)))
        _bar_w     = 28
        _frames_in = 0
        _frames_since_out = 0
        _bar_active = False   # True when a \r line is pending (needs \n to close)

        def _bar(filled, total):
            n = int(_bar_w * min(filled, total) / total)
            return '█' * n + '░' * (_bar_w - n)

        while not self.stop_ev.is_set() and not stop_ev.is_set():
            try:
                item = self.in_queue.get(timeout=1.0)
            except Empty:
                continue

            back, navi = item
            _frames_in += 1
            # Only count accumulation frames after the initial buffer is full
            if _frames_in > _n_shots:
                _frames_since_out += 1

            # ── progress bar ──────────────────────────────────────────────────
            if _frames_in <= _n_shots:
                phase = f'Buffering  [{_bar(_frames_in, _n_shots)}] {_frames_in:4d}/{_n_shots}'
            else:
                phase = f'Accumulate [{_bar(_frames_since_out, _out_times)}] {_frames_since_out:3d}/{_out_times}'
            print(f'\r  {phase}', end='', flush=True)
            _bar_active = True

            try:
                result = self.processor.update(back, navi)
                self._proc_active_time = time()
            except Exception:
                print()
                _bar_active = False
                log.exception("Processing error")
                self._emit_status(_STATE_ERROR, 0, force=True)
                self._restart_processor("processing exception")
                return

            if result["out"] is None:
                # No full result yet → emit a status heartbeat so the external
                # module sees liveness and buffer-fill progress.
                if _frames_in <= _n_shots:
                    self._emit_status(_STATE_ACCUM, _frames_in / _n_shots * 100,
                                      pulse=back.pulse, step=back.step)
                else:
                    self._emit_status(_STATE_READY, _frames_since_out / _out_times * 100,
                                      pulse=back.pulse, step=back.step)
                continue

            # ── result ready ──────────────────────────────────────────────────
            print()  # close the \r progress bar
            _bar_active = False
            _frames_since_out = 0
            o = result["out"]
            self._last_output = o   # cache for heartbeats (keep receiver's picture)
            curr_spd = getattr(o, 'curr_speed', 0.0)
            curr_dir = getattr(o, 'curr_dir',   0.0)
            print(
                f'  → Hs={o.wave_sum.swh:.2f}m  Tp={o.wave_sum.t_p:.1f}s  '
                f'Dp={o.wave_sum.d_p:.0f}°  Nsys={o.ide_sys}  '
                f'Curr={curr_spd:.2f}m/s@{curr_dir:.0f}°  '
                f'{"GOOD" if o.n_dis else "BAD"}'
            )
            log.info(
                f"Hs={o.wave_sum.swh:.2f}m  Tp={o.wave_sum.t_p:.1f}s  "
                f"Dp={o.wave_sum.d_p:.0f}°  Curr={curr_spd:.2f}m/s  "
                f"Nsys={o.ide_sys}"
            )

            proc_result = ProcessResult(
                output=result["out"],
                port=result["port"],
                navi=navi,
            )
            try:
                self.out_queue.put_nowait(proc_result)
            except Full:
                log.warning("Output queue full — result dropped")

        # Clean up any pending \r progress bar when this thread exits
        if _bar_active:
            print()
        log.warning("Process thread stopped")

    # ── output thread ─────────────────────────────────────────────────────────

    def _output_loop(self):
        log.info("Output thread started")
        while not self.stop_ev.is_set():
            try:
                result = self.out_queue.get(timeout=1.0)
            except Empty:
                continue

            for sink in self.out:
                try:
                    sink.send(result)
                except Exception:
                    log.exception(f"Sink {type(sink).__name__} send error")

        log.warning("Output thread stopped")

    # ── shutdown ──────────────────────────────────────────────────────────────

    def _shutdown(self):
        log.warning("Shutting down manager...")
        self.stop_ev.set()
        self.proc_stop_ev.set()

        for t, name in [
            (self.t_inp, "Input"),
            (self.t_proc, "Process"),
            (self.t_out, "Output"),
        ]:
            if t and t.is_alive():
                t.join(timeout=5)
                if t.is_alive():
                    log.warning(f"{name} thread did not stop in time")

        try:
            self.inp.close()
        except Exception:
            pass

        for sink in self.out:
            try:
                sink.close()
            except Exception:
                pass

        log.warning("Manager shutdown complete")
