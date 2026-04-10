from configparser import ConfigParser
from dataclasses import dataclass


@dataclass(frozen=True)
class Constants:
    AAP: int
    ARDP: int
    ADP: int
    ASP: int
    K_NUM: int
    N_SHOTS: int
    N_FREQ: int
    NUM_AREA: int
    SNR_A: int
    SNR_B: int
    MEAN: int
    CHANGE_DIR_NUM_SHOTS: int
    RPM: int


@dataclass(frozen=True)
class PipelineConfig:
    queue_size: int
    restart_on_error: bool


@dataclass(frozen=True)
class AppConfig:
    input: dict
    output: dict
    pipeline: PipelineConfig
    const: Constants


def load_config(path: str) -> AppConfig:
    cfg = ConfigParser(allow_no_value=True, inline_comment_prefixes=(";", "#"))
    cfg.read(path)

    const = Constants(
        AAP=cfg.getint("constants", "AREA_AZIM_PX"),
        ARDP=cfg.getint("constants", "AREA_READ_DIST_PX"),
        ADP=cfg.getint("constants", "AREA_DISTANCE_PX"),
        ASP=cfg.getint("constants", "AREA_SIZE_PX"),
        K_NUM=cfg.getint("constants", "K_NUM"),
        N_SHOTS=cfg.getint("constants", "N_SHOTS"),
        N_FREQ=cfg.getint("constants", "N_FREQ"),
        NUM_AREA=cfg.getint("constants", "NUM_AREA"),
        SNR_A=cfg.getint("constants", "SNR_A"),
        SNR_B=cfg.getint("constants", "SNR_B"),
        MEAN=cfg.getint("constants", "MEAN"),
        CHANGE_DIR_NUM_SHOTS=cfg.getint("constants", "CHANGE_DIR_NUM_SHOTS"),
        RPM=cfg.getint("constants", "RPM"),
    )

    pipeline = PipelineConfig(
        queue_size=cfg.getint("pipeline", "queue_size"),
        restart_on_error=cfg.getboolean("pipeline", "restart_on_error"),
    )

    return AppConfig(
        input=dict(cfg["input"]),
        output=dict(cfg["output"]),
        pipeline=pipeline,
        const=const,
    )
