from time import sleep
from src.config import load_config
from src.runtime.manager import Manager
from src.processing.processor import Processor
from src.io.input import UdpInputSource, NCInputSource
from src.io.output import UdpOutputSink, CSVOutputSink
from src.runtime.logger import setup_logger


def main():
    log = setup_logger("main")
    cfg = load_config("config.ini")
    log.info("Config loaded")

    if cfg.input.get("type", "udp") == "nc":
        source = NCInputSource(cfg.input["data_path"])
    else:
        source = UdpInputSource(cfg.input["my_ip"],
                                int(cfg.input["back_port"]),
                                int(cfg.input["navi_port"]),
                                cfg.const.AAP, cfg.const.ARDP)
    log.info("UDP rx is ready")
    sinks = []

    if cfg.output.get("udp", "false") == "true":
        sinks.append(UdpOutputSink(cfg.output["server_ip"], int(cfg.output["server_port"]),
                                   cfg.const.N_FREQ, cfg.const.NUM_AREA))

        log.info("UDP tx is ready")

    if cfg.output.get("file", "false") == "true":
        sinks.append(CSVOutputSink(cfg.output["save_path"], cfg.const.K_NUM, cfg.const.RPM,
                                   cfg.const.MEAN, cfg.const.N_SHOTS, cfg.const.N_SHOTS, cfg.const.NUM_AREA))
        log.info("File saver is ready")

    manager = Manager(config=cfg, processor_factory=lambda: Processor(cfg, cfg.output["pics"]),
                      inp_source=source, out_source=sinks)
    manager.start()
    log.info("Manager started")

    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        manager.stop()


if __name__ == "__main__":
    main()
