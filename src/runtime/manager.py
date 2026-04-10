from time import sleep
from queue import Queue, Empty, Full
from threading import Thread, Event, Lock
from src.io.input import InputSource
from src.runtime.logger import setup_logger
from src.io.output import UdpOutputSink, CSVOutputSink

log = setup_logger("manager")


class Manager:

    def __init__(self, config, processor_factory, inp_source: InputSource, out_source: list):
        self.inp = inp_source
        self.out = out_source
        self.cfg = config

        self.processor_factory = processor_factory
        self.processor = self.processor_factory()

        self.in_queue = Queue(maxsize=config.pipeline.queue_size)
        self.out_queue = Queue(maxsize=config.pipeline.queue_size)

        self.stop_ev = Event()
        self.proc_stop_ev = Event()
        self.proc_lock = Lock()

        self.t_inp = None
        self.t_out = None

    def start(self):
        log.info("Starting manager")
        self._start_processor()

        self.t_inp = Thread(target=self._input_loop, daemon=True)
        self.t_out = Thread(target=self._output_loop, daemon=True)

        self.t_inp.start()
        self.t_out.start()

    def run(self):
        self.start()
        try:
            while not self.stop_ev.is_set():
                sleep(0.1)
        except KeyboardInterrupt:
            log.warning("Keyboard interrupt received")
        finally:
            self._shutdown()

    def stop(self):
        self._shutdown()

    def _start_processor(self):
        with self.proc_lock:
            self.proc_stop_ev.clear()
            self.processor = self.processor_factory()
            self.t_proc = Thread(target=self._process_loop, daemon=True)
            self.t_proc.start()

    def _restart_processor(self, reason):
        log.error(f"Restarting processor: {reason}")

        with self.proc_lock:
            self.proc_stop_ev.set()

        sleep(0.1)

        self._clear_queue(self.in_queue)
        self._clear_queue(self.out_queue)
        self._start_processor()

    @staticmethod
    def _clear_queue(q):
        while not q.empty():
            try:
                q.get_nowait()
            except Empty:
                break

    def _input_loop(self):
        try:
            while not self.stop_ev.is_set():
                back = self.inp.get_bck()

                if back.step == 0.0:
                    log.info("Input EOF reached, stopping application")
                    self.stop_ev.set()
                    self.in_queue.put((None, None))
                    return

                navi = self.inp.get_navi()
                log.info(f"Received Navi! Lat = {navi.lat}, Lon = {navi.lon}")

                try:
                    self.in_queue.put((back, navi))
                    log.info(f"Data received: {self.processor.get_info()}")
                except Full:
                    self._restart_processor("Input queue overflow")

        except Exception:
            log.exception("Input thread crashed")
            self.stop_ev.set()

        finally:
            log.warning("Input thread stopped")

    def _process_loop(self):
        log.info("Process thread started")
        try:
            while not self.stop_ev.is_set() and not self.proc_stop_ev.is_set():
                try:
                    back, navi = self.in_queue.get()
                    if back is None and navi is None:
                        log.info("Processor received stop signal")
                        self.out_queue.put(None)
                        self.stop_ev.set()
                        return

                    result = self.processor.update(back, navi)
                    if result["out"] is not None:
                        self.out_queue.put(result)
                        log.info(f"Params updated! Hs = {result['out'].wave_sum.swh:.2f}m, "
                                 f"Tp = {result['out'].wave_sum.per:.1f}s, "
                                 f"Dp = {result['out'].wave_sum.ddir:.0f}°")

                except Empty:
                    continue

                except Full:
                    self._restart_processor("Output queue overflow")
                    return

        except Exception:
            log.exception("Processing error")
            self._restart_processor("Processing exception")
            return

        finally:
            # self.processor.stop()
            log.warning("Processor thread stopped")

    def _output_loop(self):
        try:
            while not self.stop_ev.is_set():
                try:
                    data = self.out_queue.get()
                    if data is None:
                        log.info("Output received stop signal")
                        self.stop_ev.set()
                        return
                    if data["out"] is not None:
                        for sink in self.out:
                            if isinstance(sink, UdpOutputSink):
                                sink.send(data["out"])
                            elif isinstance(sink, CSVOutputSink):
                                sink.send(data["pulse"], data["step"],
                                          data["out"].wave_sum,
                                          data["out"].spec_1d, data["port"],
                                          data["out"].spec_2d, data["navi"])
                            else:
                                pass

                except Empty:
                    continue

        except Exception:
            log.exception("Output thread error")
            self.stop_ev.set()

        finally:
            log.warning("Output thread stopped")

    def _shutdown(self):
        log.warning("Shutting down manager...")

        self.stop_ev.set()
        self.proc_stop_ev.set()

        try:
            self.in_queue.put_nowait((None, None))
        except Full:
            pass

        try:
            self.out_queue.put_nowait(None)
        except Full:
            pass

        if self.t_inp and self.t_inp.is_alive():
            self.t_inp.join(timeout=5)

        if self.t_proc and self.t_proc.is_alive():
            self.t_proc.join(timeout=5)

        if self.t_out and self.t_out.is_alive():
            self.t_out.join(timeout=5)

        try:
            self.inp.close()
        except Exception:
            pass
        #
        # for sink in self.out:
        #     try:
        #         sink.close()
        #     except Exception:
        #         pass

        log.warning("Manager shutdown complete")