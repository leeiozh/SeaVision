from configparser import ConfigParser
from dataclasses import dataclass


@dataclass(frozen=True)
class Constants:
    # ── Installation geometry (from [hardware]) ────────────────────────────────
    AAP: int
    ARDP: int
    ADP: int
    ASP: int
    RPM: int
    installation_id: str

    # ── Empirical calibration (from [calibration]) ────────────────────────────
    SNR_A: float
    SNR_B: float
    WSPD_A: float
    WSPD_B: float
    WIND_SIG_MIN: float          # min ring-std for quality=GOOD

    # ── Processing window (from [processing]) ─────────────────────────────────
    N_SHOTS: int
    MEAN: int

    # ── Algorithm & protocol version — hardcoded, NOT in config ───────────────
    # Changing these requires a matching update to tester_receive.py (N_FREQ, N_DIRS)
    # and recalibration of SNR_A/B (K_NUM). Edit here and in batch_process.py together.
    N_FREQ:    int = 64
    N_DIRS:    int = 36
    K_NUM:     int = 32
    NUM_AREA:  int = 8
    N_FREQ_2D: int = 36


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
        # [hardware]
        AAP             = cfg.getint  ("hardware",    "AREA_AZIM_PX",       fallback=4096),
        ARDP            = cfg.getint  ("hardware",    "AREA_READ_DIST_PX",  fallback=2048),
        ADP             = cfg.getint  ("hardware",    "AREA_DISTANCE_PX",   fallback=1192),
        ASP             = cfg.getint  ("hardware",    "AREA_SIZE_PX",       fallback=192),
        RPM             = cfg.getint  ("hardware",    "RPM",                fallback=25),
        installation_id = cfg.get     ("hardware",    "installation_id",    fallback="default"),
        # [calibration]
        SNR_A           = cfg.getfloat("calibration", "SNR_A",              fallback=33.0),
        SNR_B           = cfg.getfloat("calibration", "SNR_B",              fallback=36.0),
        WSPD_A          = cfg.getfloat("calibration", "WSPD_A",             fallback=0.0),
        WSPD_B          = cfg.getfloat("calibration", "WSPD_B",             fallback=40.0),
        WIND_SIG_MIN    = cfg.getfloat("calibration", "WIND_SIG_MIN",       fallback=5.5),
        # [processing]
        N_SHOTS         = cfg.getint  ("processing",  "N_SHOTS",            fallback=256),
        MEAN            = cfg.getint  ("processing",  "MEAN",               fallback=4),
        # N_FREQ, N_DIRS, K_NUM, NUM_AREA — dataclass defaults (algorithm version)
    )

    pipeline = PipelineConfig(
        queue_size      = cfg.getint    ("pipeline", "queue_size",       fallback=4),
        restart_on_error= cfg.getboolean("pipeline", "restart_on_error", fallback=True),
    )

    return AppConfig(
        input   = dict(cfg["input"])   if cfg.has_section("input")  else {},
        output  = dict(cfg["output"])  if cfg.has_section("output") else {},
        pipeline= pipeline,
        const   = const,
    )
