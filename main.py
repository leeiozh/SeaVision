import sys
import ctypes
from time import sleep
from src.config import load_config
from src.runtime.manager import Manager
from src.processing.processor import Processor
from src.io.input import UdpInputSource, NCInputSource, BT8InputSource
from src.io.output import UdpOutputSink, CSVOutputSink
from src.runtime.logger import setup_logger


def _disable_quick_edit():
    """Disable Windows console Quick Edit Mode.

    In Quick Edit Mode (default for cmd.exe) any mouse click or accidental
    keypress pauses the process until Enter is pressed — fatal for unattended
    production runs.  No-op on non-Windows platforms.
    """
    if sys.platform != "win32":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-10)          # STD_INPUT_HANDLE
        mode = ctypes.c_ulong()
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        mode.value &= ~0x0040                        # clear ENABLE_QUICK_EDIT_MODE
        mode.value |= 0x0080                         # set ENABLE_EXTENDED_FLAGS
        kernel32.SetConsoleMode(handle, mode)
    except Exception:
        pass


def _build_source(cfg):
    t = cfg.input.get("type", "udp")
    if t == "nc":
        return NCInputSource(cfg.input["data_path"])
    if t == "bt8":
        return BT8InputSource(
            folder_path=cfg.input["bt8_folder"],
            aap=cfg.const.AAP,
            ardp=cfg.const.ARDP,
            start_ind=int(cfg.input.get("bt8_start", 0)),
            end_ind=int(cfg.input.get("bt8_end", 100000)),
            pulse=int(cfg.input.get("bt8_pulse", 2)),
        )
    return UdpInputSource(
        cfg.input["my_ip"],
        int(cfg.input["back_port"]),
        int(cfg.input["navi_port"]),
        cfg.const.AAP,
        cfg.const.ARDP,
    )


def _build_sinks(cfg):
    sinks = []
    if cfg.output.get("udp", "false") == "true":
        proto = cfg.output.get("protocol", "new")
        sinks.append(UdpOutputSink(
            cfg.output["server_ip"],
            int(cfg.output["server_port"]),
            cfg.const.N_FREQ,
            cfg.const.N_DIRS,
            cfg.const.N_FREQ_2D,
            cfg.const.ALGO_VERSION,
            proto,
        ))
        ip2   = cfg.output.get("server_ip2",   "").strip()
        port2 = cfg.output.get("server_port2", "").strip()
        if ip2 and port2:
            sinks.append(UdpOutputSink(
                ip2, int(port2),
                cfg.const.N_FREQ,
                cfg.const.N_DIRS,
                cfg.const.N_FREQ_2D,
                cfg.const.ALGO_VERSION,
                cfg.output.get("protocol2", proto),
            ))
    if cfg.output.get("file", "false") == "true":
        sinks.append(CSVOutputSink(cfg.output["save_path"], cfg.const))
    return sinks


def main():
    _disable_quick_edit()
    log = setup_logger("main")
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.ini"
    cfg = load_config(config_path)
    log.info(f"Config: {config_path}")

    source = _build_source(cfg)
    log.info(f"Input: {type(source).__name__}")

    sinks = _build_sinks(cfg)
    for s in sinks:
        log.info(f"Output: {type(s).__name__}")

    manager = Manager(
        config=cfg,
        processor_factory=lambda: Processor(cfg, cfg.output.get("pics", "false")),
        inp_source=source,
        out_sinks=sinks,
    )
    manager.start()
    log.info("Manager started")

    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        manager.stop()


if __name__ == "__main__":
    main()
