# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Что это

Система обработки морского радара для оценки параметров волнения.
Радар вращается (RPM=25), каждый оборот = один кадр бэкскаттера `(AAP=4096, ARDP=2048)` пикселей.
Обработка в реальном времени: UDP-вход → процессор → UDP-выход → tester_receive.py (визуализация).

## Команды

```bash
# Активация виртуального окружения (Python 3.12)
source sv_env/bin/activate

# Запуск основного процессора
python main.py                   # использует config.ini по умолчанию
python main.py config_debug.ini  # явный путь к конфигу (NC-источник, N_SHOTS=64, pics=./)

# Визуализация выхода (отдельный терминал, UDP порт 4000)
python test/tester_receive.py

# Генерация тестового UDP-потока для отладки без радара
python test/tester_transmit.py

# Пакетная обработка NetCDF-файлов
python batch_process.py [--csv META_upd2.csv] [--base-path /storage/thalassa/DATA/RADAR/] [--out batch_out] [--config config.ini]

# Параллельная пакетная обработка (multiprocessing или SLURM)
python batch_process_parallel.py --n-workers 8
python batch_process_parallel.py --task-id $SLURM_ARRAY_TASK_ID --n-tasks $SLURM_ARRAY_TASK_COUNT
python batch_process_parallel.py --merge-only --out batch_out

# Сборка Windows .exe через GitHub Actions
# → push в origin/main или ручной запуск workflow "Build Windows EXE"
# → скачать артефакт seavision-windows-x64 → положить config.ini рядом с seavision.exe

```

Нет системы тестов и линтера — тестирование через `tester_receive.py` с живыми данными или файлами.

`test/tester_receive.py` и `test/tester_transmit.py` — канонические.

## Архитектура потока данных

```
UdpInputSource / NCInputSource / BT8InputSource
    ↓  back (BackData), navi (Navi)
Manager (три потока: Input → Process → Output)
    ↓
Processor.update()          ← основная логика
    ↓
Averager.push() → Averager.get_mean() → Output
    ↓
UdpOutputSink / CSVOutputSink
    ↓  UDP пакет 1412 байт  /  4 CSV файла с timestamp
test/tester_receive.py      ← визуализация
```

`out_times` в `[output]` — каждые сколько кадров вызывать расчёт спектра (default 32). Расчёт запускается при `s.index >= N_SHOTS and s.index % out_times == 0`.

## Manager: модель устойчивости

| Событие | Реакция |
|---|---|
| Input timeout / EOF | silent wait; после `N_SHOTS × rot_period` с молчания → `_reset_pending = True` |
| Data resumed | перезапуск processor перед подачей новых данных |
| Processing exception | немедленный `_restart_processor()` |
| Processor hang > `10 × rot_period` | watchdog → `_restart_processor()` |
| Output sink error | логируется per-sink, остальные sinks продолжают работу |

## Источники входных данных

| Тип (config `type=`) | Класс | Описание |
|---|---|---|
| `udp` | `UdpInputSource` | Живой радар. Принимает пакеты по 1032 байт, собирает AAP строк → один `BackData`. **`get_bck()` возвращает копию** `bck`-массива (защита от гонки данных между Input и Process потоками). |
| `nc` | `NCInputSource` | NetCDF-файл. `back.step = 0.0` = признак EOF. |
| `bt8` | `BT8InputSource` | Папка с бинарными BT8-файлами. |

`UdpInputSource` таймауты: `overall_timeout=30` с, `per_recv_timeout=2` с.
`curr_navi` инициализируется дефолтным `Navi(0,0,0,0,0,0)` — не `None`.

**Единицы нави-данных:** `sog` и `spd` — в **м/с**, `hdg`, `cog` — в градусах. Не конвертировать.

**COG в processor.py** — circular mean через `arctan2(mean(sin), mean(cos))` по буферу `s.cog` размера N_SHOTS.

## Ключевые алгоритмы

### Методология обработки

**Шаги 1–2: накопление спектра**

1. Все NUM_AREA сегментов одновременно, без поворота. Каждый — квадрат `2·ASP × 2·ASP` px, центр на `ADP` px.
2. 3D Welch FFT каждого сегмента → `spec_3d_i (N_SHOTS//2, 2·K_NUM, 2·K_NUM)`, усреднение по NUM_AREA.

**Шаги 3–4: идентификация систем и оценка тока**

3. Pre-analysis: `apply_doppler_3d_vec(spec_3d_corr, Ux_ship, Uy_ship)` → `spec_3d_ship` → `find_freq_peaks` → `find_system_dirs` → `systems_draft`.
4. Ток: если `len(systems_draft) >= 2` → `calc_current_multiwave`; иначе → `calc_current_vector`. Клип: `_MAX_CURRENT = 3.0 м/с` и `_MULTI_MAX_CURRENT = 3.0 м/с`.

**Шаги 5–11: финальный спектр и разбиение**

5. `apply_doppler_3d_vec(spec_3d_corr, k_max, Ux, Uy, om_max)` → `spec_3d_fixed`.
6. `calc_port(spec_3d_fixed)` → `port_fixed`.
7. `separate_signal_noise`: полоса `±_SIGNAL_BAND=10 бинов` вокруг `ω = √(g·k)`.
8. MTF: `k^{-1.2}`.
9. 1D спектр → `T_peak`, `T_mean`, `m0`, `snr_tot`.
10. `calc_spec2d(spec_3d_fixed, ...)` → `s_om_th`, `peak_dir`, `mean_dir`.
11. `calc_partitions` → системы. Каждая система: `{h_s, t_p, d_p}` — **только пиковые значения**, средние не вычисляются. `wave_sum.t_m` и `wave_sum.d_m` вычисляются из 1D спектра и `calc_spec2d`, не из partitioning.

### Ток (течение)

`(Ux, Uy)` — кажущаяся скорость, **kx=Восток, ky=Север**.

```
u_curr_East  = Ux − SOG · sin(COG)
u_curr_North = Uy − SOG · cos(COG)
curr_speed   = hypot(u_curr_East, u_curr_North)   # клип 3.0 м/с
curr_dir     = degrees(arctan2(u_curr_East, u_curr_North)) % 360   # КОМПАС
```

→ `curr_speed` в UDP поле [18] uint16 ×100.
→ `curr_dir` в UDP поле [19] целые градусы.

### Скорость ветра
```
wspd = 0.01 · (WSPD_A + WSPD_B · ring_sig)   [м/с]
```
`ring_sig` = std бэкскаттера в кольце `ADP±ASP`.

### SWH и SNR
```
snr_tot        = ∬ signal_mtf dω dk  /  ∬ noise dω dk
wave_sum.swh   = 0.01 · (SNR_A + SNR_B · √snr_tot)   [м]
h_s_i          = swh · √(frac_i / Σfrac_j)            # Σh_s_i² = swh²
```

### Averager
Кольцевой буфер `MEAN` выходов. `get_mean()` возвращает `(Output, port)` — 2-tuple (spec_1d/spec_2d только внутри Output). Спектральные массивы нормированы в [0, 255].

`matplotlib` в `processor.py` импортируется **лениво** — только внутри `_save_debug_*` функций. При `pics=false` matplotlib не нужен и в бандл не включается.

## Основные структуры

### Wave
```python
Wave(swh, snr, t_p, t_m, d_p, d_m)
# t_p, d_p — пиковые; t_m, d_m — средние (только для wave_sum; для систем = 0)
```

### ProcessorState
```python
index: int
cbck: (NUM_AREA, N_SHOTS, 2·ASP, 2·ASP)  # float32
speed, heading, cog: (N_SHOTS,) / (MEAN,) / (N_SHOTS,)
curr_step: float
curr_pulse: int
```

### Processor.update() return → ProcessResult
```python
{"out": Output | None, "pulse": int, "step": float, "navi": Navi, "port": ndarray | None}
ProcessResult(output: Output, port: ndarray, navi: Navi)
```

### UDP пакет v2.0 — **1412 байт** (< 1472, без IP-фрагментации)

```
"<BBHHHHHHHHHHHHHHHHHHHHBBHHHH{N_FREQ}B{N_FREQ_2D×N_DIRS}B"
 52 байта заголовок + 64 байта spec_1d + 36×36 байт spec_2d
```

| Индекс | Тип | Поле | Кодирование |
|---|---|---|---|
| [0] | B | type=5 | — |
| [1] | B | pulse | 1/2/3 |
| [2] | H | step_mm | м×1000 |
| [3] | H | rpm_x100 | об/мин×100 |
| [4] | H | swh_sum | м×100 |
| [5] | H | t_p_sum | с×100 |
| [6] | H | t_m_sum | с×100 |
| [7] | H | dir_p_sum | °, целые |
| [8] | H | dir_m_sum | °, целые |
| [9..11] | H×3 | wind: swh, t_p, dir_p | м×100 / с×100 / ° |
| [12..14] | H×3 | sw1: swh, t_p, dir_p | аналогично |
| [15..17] | H×3 | sw2: swh, t_p, dir_p | аналогично |
| [18] | H | curr_speed | м/с×100 |
| [19] | H | curr_dir | °, КОМПАС |
| [20] | H | wspd_x10 | м/с×10 |
| [21] | H | wind_dir | °, математическая конвенция |
| [22] | B | n_sys | 0–3 |
| [23] | B | quality | 0=BAD, 1=GOOD |
| [24] | H | algo_version | `Constants.ALGO_VERSION`; инкрементировать при смене поведения алгоритма |
| [25..27] | H×3 | reserved | =0 |
| [28..91] | B×64 | spec_1d | [0–255] |
| [92..] | B×1296 | spec_2d 36×36 | row-major: dir×freq |

**Конвенции направлений:**
- `dir_p/m_sum`, `dir_p_win/sw1/sw2`, `wind_dir` — **математическая**: 0°=Восток, CCW
- `curr_dir` — **компасная**: 0°=Север, CW; `arctan2(East, North) % 360`
- `spec_2d` строка i — математический угол i×10°

**Авторитетный источник протокола**: `udp_protocol.docx` (v2.0).

### CSVOutputSink
При `file = true` создаёт 4 файла. Имя: `{installation_id}_{YYYYMMDDTHHMMSS}_*.csv` (если `installation_id != "default"`), иначе просто `{timestamp}_*.csv`.
- `_params.csv` — `datetime;pulse;step;swh;t_p;d_p;d_m;t_m;freq[0..N_FREQ-1]`
- `_port.csv` — ω-k портрет `(N_SHOTS//2, K_NUM)`
- `_spec.csv` — направленный спектр `(N_DIRS, N_FREQ_2D)` = 36×36
- `_navi.csv` — `datetime,lat,lon,spd,sog,cog,hdg`

Файловые дескрипторы держатся открытыми (`buffering=1`) — не открываются/закрываются на каждую запись.

## Файлы проекта

| Файл | Назначение |
|------|-----------|
| `src/processing/processor.py` | Главный процессор; `_SIGNAL_BAND=10`; `_MAX_CURRENT=3.0`; debug-функции с ленивым импортом matplotlib |
| `src/processing/state.py` | `ProcessorState` |
| `src/processing/averaging.py` | `Averager(mean, n_freq, n_freq_2d, n_dirs, n_shots, cut_num)` — кольцевой буфер, нормировка в [0,255]; возвращает `(Output, port)` |
| `src/algorithms/spectrum2d.py` | `calc_spec3d`, `calc_port`, `apply_doppler_3d_vec`, `separate_signal_noise`, `apply_mtf`, `compute_snr`, `compute_frequency_spectrum`, `calc_spec2d` |
| `src/algorithms/dispersion.py` | `calc_current_vector`, `calc_current_multiwave`; `_MULTI_MAX_CURRENT=3.0` |
| `src/algorithms/partition.py` | `calc_wspd`, `find_freq_peaks`, `find_system_dirs`, `calc_partitions`; системы возвращают только `{h_s, t_p, d_p}` |
| `src/algorithms/area.py` | `Area.calc_mask()` |
| `src/config.py` | `load_config()` с fallback-значениями; N_FREQ/N_DIRS/K_NUM/NUM_AREA/N_FREQ_2D — захардкожены |
| `src/io/structs.py` | Все структуры данных |
| `src/io/output.py` | `UdpOutputSink`, `CSVOutputSink` |
| `src/io/input.py` | `UdpInputSource` (get_bck возвращает copy), `NCInputSource`, `BT8InputSource` |
| `src/io/service.py` | UDP-сокеты |
| `src/runtime/manager.py` | Трёхпоточный пайплайн с watchdog |
| `src/runtime/logger.py` | `setup_logger()` — настройка форматированного лога |
| `config.ini` | Продакшн UDP (pics=false, type=udp) |
| `config_debug.ini` | Отладка (type=nc, N_SHOTS=64, pics=./, file=true) |
| `config_udp.ini` | Синоним config.ini (источник для копирования) |
| `test/tester_receive.py` | Визуализация UDP v2.0: 1D спектр + полярный спектр + таблица |
| `test/tester_transmit.py` | Генератор тестового потока |
| `seavision-win.spec` | PyInstaller onedir для Windows (без matplotlib, без UPX) |
| `seavision.spec` | PyInstaller spec для локальной сборки (Linux/общий) |
| `requirements-win.txt` | numpy, scipy, netCDF4, cftime — минимум для Windows-бандла |
| `.github/workflows/build-windows.yml` | GHA: Windows x64 → артефакт `seavision-windows-x64` |
| `udp_protocol.docx` | **Авторитетный** протокол v2.0 |
| `batch_process.py` | Пакетная обработка; `_MAX_CURRENT=3.0`, `_SIGNAL_BAND=10` |
| `batch_process_parallel.py` | Параллельная обёртка (multiprocessing / SLURM) |

### Устаревшие файлы
`src/algorithms/portrait.py`, `src/algorithms/direction.py` — не импортируются. Не трогать.

## Конфигурация

Секции и их назначение:

```
[hardware]        ← геометрия инсталляции, задаётся при монтаже
  installation_id  AREA_AZIM_PX  AREA_READ_DIST_PX  AREA_DISTANCE_PX  AREA_SIZE_PX  RPM

[calibration]     ← подгоняется по месту после установки
  SNR_A  SNR_B    — SWH = 0.01·(SNR_A + SNR_B·√snr)
  WSPD_A  WSPD_B  — WSPD = 0.01·(WSPD_A + WSPD_B·ring_sig)
  WIND_SIG_MIN    — минимальный std кольца для quality=GOOD (зависит от инсталляции)

[processing]      ← N_SHOTS можно снижать при отладке
  N_SHOTS=256  MEAN=4

[input] / [output] / [pipeline]  ← сетевые настройки, файлы, очередь
```

**Захардкожены в `load_config()`, не в config.ini** (версия алгоритма/протокола):
```
N_FREQ=64  N_DIRS=36  K_NUM=32  NUM_AREA=8  N_FREQ_2D=36  ALGO_VERSION=1
```

Флаги качества (**захардкожены в коде**, одинаковы в `processor.py` и `batch_process.py`):

| Параметр | Значение | Место |
|---|---|---|
| `_SNR_QUALITY_MIN` | 5.0 | processor.py |
| `WIND_SIG_MIN` | 5.5 (default) | **config [calibration]** |
| `_T_PEAK_MIN` | 5.5 | processor.py |

`quality = 1` если все условия выполнены И `n_sys >= 1`.

`om_max = π · RPM / 60 ≈ 1.309 rad/s`.

## Алгоритмические пороги

### `src/algorithms/partition.py`

| Константа | Значение | Назначение |
|---|---|---|
| `_FPEAK_MIN_PROM_REL` | 0.15 | Минимальная высота пика [доля от max] |
| `_FPEAK_BAND_HALF_FRAC` | 0.10 | Полуширина полосы пика [доля от n_om] |
| `_FPEAK_MIN_SEP_FRAC` | 0.12 | Минимальное расстояние между пиками |
| `_FPEAK_SMOOTH_SIGMA` | 2.0 | Гауссово сглаживание [бин] |
| `_FPEAK_MAX` | 3 | Максимальное число пиков |
| `_SDIR_SMOOTH_SIGMA` | 1.0 | Сглаживание угловой проекции [бин] |
| `_PART_MIN_DIR_SEP` | 25.0 | Минимальное угловое расстояние [°] |
| `_PART_MIN_PER_RATIO` | 1.1 | T_large/T_small для различимости |
| `_PART_MIN_ENERGY_FRAC` | 0.05 | Минимальная доля энергии пика |
| `_PART_NOISE_SNR` | 3.0 | Пик > медиана × этот коэффициент |
| `_PART_BLANK_FRAC` | 0.08 | Радиус гашения [доля оси] |
| `_PART_WIND_DIR_THRESH` | 45.0 | Максимальный угол от wdir для ветровой [°] |

### `src/algorithms/dispersion.py`

| Константа | Значение | Назначение |
|---|---|---|
| `_MULTI_DIR_HALF_DEG` | 45.0 | Угловой конус вокруг направления системы [°] |
| `_MULTI_MIN_CELLS` | 5 | Минимум ячеек для МНК |
| `_MULTI_MIN_SV_RATIO` | 0.05 | Защита от вырожденной матрицы |
| `_MULTI_MAX_CURRENT` | **3.0** | Клип остаточного тока в multiwave [м/с] |
| `_MULTI_K_MIN_REL` | 0.08 | Минимальный k/k_max |

### `src/processing/processor.py`

| Константа | Значение | Назначение |
|---|---|---|
| `_SIGNAL_BAND` | 10 | Полуширина дисперсионной полосы [k-бинов]; при изменении перекалибровать SNR_A/B |
| `_MAX_CURRENT` | **3.0** | Клип скорости тока перед выводом [м/с] |
| `_SNR_QUALITY_MIN` | 5.0 | Минимальный SNR для quality=GOOD |
| `_T_PEAK_MIN` | 5.5 | Минимальный T_peak [с] для quality=GOOD |

## Debug-выводы

`Processor(config, pics)` — `pics=False`/`"false"` отключают; строка с путём — включают.
PNG `debug_combined.png`: [0,0] 1D спектр; [0,1] полярный спектр; [0,2] ω-k портрет; [1,:] таблица параметров.

## Визуализация (tester_receive.py)

- `F_DISPLAY = 0.20` Hz — радиальный предел
- `_RENDER_DT = 0.35 с` — декаплинг UDP/отрисовки
- spec_2d приходит как (36, 36) row-major; `.T` для pcolormesh

## Что НЕ нужно делать

- **Не применять** `apply_doppler_2d` и скалярный `apply_doppler_3d` — устарели.
- **Не прибавлять SOG к `(Ux, Uy)` перед Doppler-коррекцией** — они уже включают скорость судна.
- **При multiwave**: `calc_current_multiwave` принимает `spec_3d_ship`, возвращает остаточный `(Ucx, Ucy)`; `Ux = Ucx + Ux_ship`.
- **Не использовать** удалённые поля Wave: `per`, `len`, `dir`, `ddir`, `vco`, `inv`.
- **При изменении `_SIGNAL_BAND`** — перекалибровать SNR_A/B синхронно в `processor.py` и `batch_process.py`.
- **Не трогать** `udp_protocol.docx` вручную — он генерируется скриптом.
- **Не реанимировать** `portrait.py`, `direction.py`.
- **Averager.get_mean()** возвращает 2-tuple `(Output, port)`, не 4-tuple.
- **calc_partitions** возвращает на систему только `{h_s, t_p, d_p}` — без `t_m`, `d_m`.

## Ссылки на литературу

Допплер-коррекция: Carrasco, Lund, Nieto-Borge, Young. Принцип: `Δω = kx·Ux + ky·Uy`.
Мультиволновой ток: Dankert & Rosenthal 2004; Stewart & Joy 1974. Fallback при коллинеарных системах.
