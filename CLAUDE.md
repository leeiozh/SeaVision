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

# Запуск основного процессора (конфиг захардкожен в main.py как "config.ini")
python main.py

# Визуализация выхода (отдельный терминал, UDP порт 4000)
python test/tester_receive.py

# Генерация тестового UDP-потока для отладки без радара
python test/tester_transmit.py

# Пакетная обработка NetCDF-файлов (без усреднения, первые N_SHOTS кадров каждого)
python batch_process.py [--csv META_upd.csv] [--base-path /storage/thalassa/DATA/RADAR/] [--out batch_out] [--config config.ini]
# files.csv: одна колонка 'name' с относительными путями; --base-path — префикс к каждому пути
# вывод: {out}/params.csv + {out}/spec/{name}_freqspec.npy / _dirspec.npy + {out}/pics/{name}.png

# Параллельная пакетная обработка (multiprocessing или SLURM)
python batch_process_parallel.py --n-workers 8   # локальный режим
python batch_process_parallel.py --task-id $SLURM_ARRAY_TASK_ID --n-tasks $SLURM_ARRAY_TASK_COUNT  # SLURM
python batch_process_parallel.py --merge-only --out batch_out   # слияние partial CSV после SLURM
# run_batch.sh — SLURM sbatch-скрипт (48 воркеров, партиция r2c2)

# Сборка в .exe (PyInstaller)
pyinstaller main.py --onedir
```

Нет системы тестов и линтера — тестирование через `tester_receive.py` с живыми данными или файлами.

`test/tester_receive.py` и `test/tester_transmit.py` — канонические; в корне репозитория лежат их копии (не синхронизируются).

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
    ↓  UDP пакет 2420 байт  /  4 CSV файла с timestamp
test/tester_receive.py      ← визуализация
```

`out_times` в `[output]` — каждые сколько кадров вызывать расчёт спектра (default 32). Расчёт запускается при `s.index >= N_SHOTS and s.index % out_times == 0`; до этого `update()` только заполняет буфер `cbck`. `Averager` усредняет до последних `MEAN` расчётов; вывод начинается с первого накопленного результата (не ждёт MEAN).

## Manager: модель устойчивости

Три потока: Input → Process → Output, связаны через `in_queue` и `out_queue`.

| Событие | Реакция |
|---|---|
| Input timeout / EOF | silent wait; после `N_SHOTS × rot_period` с молчания → `_reset_pending = True` |
| Data resumed | перезапуск processor перед подачей новых данных, `_reset_pending = False` |
| Processing exception | немедленный `_restart_processor()` |
| Processor hang > `10 × rot_period` | watchdog → `_restart_processor()` |
| Output sink error | логируется per-sink, остальные sinks продолжают работу |

При перезапуске processor: drain `in_queue`, пересоздать `Processor` через `processor_factory`, запустить новый `t_proc`.

## Источники входных данных

| Тип (config `type=`) | Класс | Описание |
|---|---|---|
| `udp` | `UdpInputSource` | Живой радар. Принимает пакеты по 1032 байт, собирает AAP строк → один `BackData`. Дублирование строк (`double_counter ≥ 4`) принудительно завершает сбор кадра. |
| `nc` | `NCInputSource` | NetCDF-файл с историческими данными (путь: `data_path`). |
| `bt8` | `BT8InputSource` | Папка с бинарными BT8-файлами (путь: `bt8_folder`, индексы `bt8_start`/`bt8_end`, код импульса `bt8_pulse`). |

`UdpInputSource.get_bck()` таймаут: `overall_timeout=30` с, `per_recv_timeout=2` с на каждый recv.

`NCInputSource` возвращает `back.step = 0.0` при исчерпании файла — это признак конца данных. В `batch_process.py` range resolution захардкожен как `step = 1.875` м/пкс; `UdpInputSource` и `BT8InputSource` берут `step` из пакета.

**Единицы нави-данных:** `sog` и `spd` приходят в raw-пакете в см/с (знаменатель 100), в коде — **м/с**, конвертировать не нужно. `hdg`, `cog` — градусы (знаменатель 100).

## Ключевые алгоритмы

### Методология обработки (актуальная, после рефакторинга мая 2026)

1. **Все NUM_AREA сегментов одновременно, без поворота** (`orient=0`).
   Каждый сегмент — квадрат `2·ASP × 2·ASP` пикселей, центр на расстоянии `ADP` px под своим азимутом.
   Азимуты: `np.linspace(0, 360, NUM_AREA, endpoint=False)`.

2. **3D Welch FFT каждого сегмента** → `spec_3d_i (N_SHOTS//2, 2·K_NUM, 2·K_NUM)`, усреднение по NUM_AREA → `spec_3d_corr`.

3. **Оценка вектора тока `(Ux, Uy)`** из `spec_3d_corr` через `calc_current_vector`:
   для каждой валидной ячейки `(kx, ky)` — энергетический центроид ω, уравнение `ω̄ − √(g|k|) = kx·Ux + ky·Uy`, МНК по всем ячейкам.
   `(Ux, Uy)` = кажущаяся скорость = ток_воды − скорость_судна, в географическом фрейме (Север, Восток).
   Двухпроходная оценка: Pass 1 — **argmax** в широком окне `max(_SIGNAL_BAND, n_om//4)`, инициализация от скорости судна `(Ux_prior = −SOG·sin(COG), Uy_prior = −SOG·cos(COG))` — окно сразу над областью энергии; Pass 2 — **центроид** в узком окне `_SIGNAL_BAND` вокруг результата Pass 1 (субпиксельная точность).

4. **Векторная Доплеровская коррекция** `apply_doppler_3d_vec(spec_3d_corr, k_max, Ux, Uy, om_max)`:
   каждая ячейка `(kx, ky)` сдвигается на `Δω = kx·Ux + ky·Uy` — точно для любого числа волновых систем.
   → `spec_3d_fixed`.

5. **ω-k портрет** `port_fixed` из `spec_3d_fixed` через `calc_port`.

6. **Разделение сигнал/шум**: полоса `±_SIGNAL_BAND=10 бинов` вокруг `ω = √(g·k)` — единый параметр для всех шагов.

7. **MTF коррекция**: `k^{-1.2}`.

8. **1D ω-спектр** из MTF-взвешенного сигнала → `T_peak`, `T_mean`, `m0`, `snr_tot`.

9. **Направленный спектр** `s_om_th (N_DIRS, N_SHOTS//2)` из `spec_3d_fixed` через `calc_spec2d` с тем же `band=_SIGNAL_BAND`.
    Бин 0 = направление вдоль оси +X радарного изображения (= Север, т.к. изображение всегда север вверх).

10. **Разбиение на системы** `calc_partitions` — итеративный поиск пиков + гасение (радиус 15% по ω и dir).
    Стоп-критерии: пик < 5× медиана фона, оставшаяся энергия < 20% суммарной.
    Фильтр per-системы: `sys_energy / total_energy < MIN_ENERGY_FRAC (0.20)` → система отбрасывается.
    Слияние дубликатов: два условия независимы (OR): пик пропускается если `dir_sep < MIN_DIR_SEP (40°)` ИЛИ `T_large / T_small < MIN_PER_RATIO (1.3)`.
    Классификация: система в пределах ±45° от wdir (кратчайший T среди кандидатов) = ветровая; нет кандидата → w_s=None, все системы = зыби (по возрастанию T).
    Сохранение энергии: raw fracs нормируются (`frac_i / Σfrac_j`) → `h_s_i = swh·√(frac_norm_i)`, гарантирует `Σ h_s_i² = swh²`.

### Ток (течение)
Координатная система сегмента (`orient=0`): строки (axis 0) идут с Запада на Восток, столбцы (axis 1) — с Юга на Север. После FFT:
- **kx (axis 0) = Восток**, ky (axis 1) = Север

`(Ux, Uy)` из `calc_current_vector` — кажущаяся скорость в координатах **(Восток, Север)**.

Истинный ток воды:
```
u_curr_East  = Ux + SOG · sin(COG)   # Восток
u_curr_North = Uy + SOG · cos(COG)   # Север
curr_speed   = hypot(u_curr_East, u_curr_North)
curr_dir     = degrees(arctan2(u_curr_East, u_curr_North)) % 360
```
→ `curr_speed` упаковывается в UDP слот `un[11]` как uint8 (`× 100`, clip 0–255 cm/s).
→ `curr_dir` упаковывается в UDP слот `un[7]` в целых градусах.

HDG не нужен: изображение не привязано к курсу судна.

`SOG` и `COG` из `navi` — в **м/с** и **градусах** соответственно. Не умножать на коэффициенты пересчёта.

### Скорость ветра
```
wspd = 0.01 · (WSPD_A + WSPD_B · ring_sig)   [м/с]
```
`ring_sig` = std интенсивности бэкскаттера в кольце `ADP±ASP`. `WSPD_A`, `WSPD_B` — в `[constants]` конфига (множитель 0.01 в коде, поэтому `WSPD_A≈149` → ~1.49 м/с).

### SWH и SNR

```
port_fixed = omega-k портрет Doppler-скорректированного spec_3d_fixed
signal     = port_fixed, |ω_bin − ω_ref(k)| ≤ _SIGNAL_BAND
noise      = port_fixed, |ω_bin − ω_ref(k)| >  _SIGNAL_BAND
signal_mtf = signal · k^{-1.2}

snr_tot    = ∬ signal_mtf dω dk  /  ∬ noise dω dk

wave_sum.swh = 0.01 · (SNR_A + SNR_B · √snr_tot)   [м]
```

SNR_A и SNR_B подлежат перекалибровке при изменении band или схемы коррекции.

Sub-системы в `calc_partitions`:
```
frac_raw_i  = sys_energy_i / total_energy          # до зануления пика
frac_norm_i = frac_raw_i / Σ frac_raw_j            # нормировка → Σ frac_norm = 1
h_s_i       = wave_sum.swh · √frac_norm_i          # Σ h_s_i² = swh²
```

### Направление ветра
`calc_wspd(bck)` — аппроксимация `I(θ) = a + b·cos²(0.5·(θ−c))` по азимутальной средней интенсивности.
Возвращает `(sig, wdir_deg)`. `wdir` передаётся в `calc_partitions()` только как ориентир классификации (система в пределах ±60° от wdir = ветровая); `wave_win.d_p` — спектральный пик ветровой системы, а не `wdir` напрямую.

### Averager
Хранит кольцевой буфер последних `MEAN` выходов. `get_mean()` усредняет их (до `min(index, MEAN)` накоплений) и **нормирует спектральные массивы в [0, 255]** перед возвратом. Возвращает `(Output, spec_1d, spec_2d, port)` — все spectral arrays нормированы в `[0, 255]` и приведены к `int`; поля Wave (swh, t_p и т.д.) нормировке не подвергаются.

В `processor.update()` возвращённые `spec_1d`/`spec_2d` (позиции 2,3) не используются; `port` (позиция 4) идёт в `ProcessResult.port` → `_port.csv` sink.

## Основные структуры

### Wave
```python
Wave(swh, snr, t_p, t_m, d_p, d_m)
# t_p = пиковый период [с]
# t_m = средний период [с]
# d_p = пиковое направление [°]
# d_m = среднее направление [°]
```
Удалённые поля (не использовать): `vco`, `inv`, `per`, `len`, `dir`, `ddir`.

### WaveOutput
Промежуточная структура, передаётся из `Processor.update()` в `Averager.push()` (в sinks не попадает):
```python
WaveOutput(ide_sys, wave_sum, wave_win, wave_sw1, wave_sw2, spec_1d, spec_2d)
```
`Averager.push()` накапливает их в кольцевом буфере; `get_mean()` строит полноценный `Output`.

### Output
Центральная структура, создаётся `Averager.get_mean()` и передаётся через `ProcessResult.output` в sinks:
```python
Output(
    pulse, step, rps,          # параметры радара
    n_in_win, n_wins,          # кол-во кадров в окне и накоплений
    step_area, n_area,         # шаг и размер сегмента
    n_start,                   # переиспользован под HDG (целые градусы)
    cog_proc, sog_proc,        # навигация [м/с], заполняется в processor.update()
    max_sys, ide_sys,          # макс. и текущее число систем
    wave_sum, wave_win,        # суммарная и ветровая волны (Wave)
    wave_sw1, wave_sw2,        # зыби 1 и 2 (Wave)
    n_dis,                     # legacy поле
    spec_1d,                   # нормированный 1D спектр [0..255], int array
    spec_2d,                   # нормированный 2D спектр [0..255], int array
)
# Дополнительные поля, устанавливаемые в processor.update() после get_mean():
# .curr_speed — модуль истинного тока воды [м/с]
# .curr_dir   — компасный курс тока [°]
# .wind_dir   — направление ветра по бэкскаттеру [°]
# .wspd       — скорость ветра [м/с] = WSPD_A + WSPD_B × ring_sig
```

### ProcessorState
```python
index: int                                # монотонный счётчик кадров; триггер расчёта: index >= N_SHOTS and index % out_times == 0
cbck: (NUM_AREA, N_SHOTS, 2·ASP, 2·ASP)  # float32, 4D буфер всех сегментов
speed, heading, cog: (MEAN,)              # скользящие окна нави-данных [м/с и °]
curr_step: float                          # range resolution [м/px] текущего кадра
curr_pulse: int                           # код импульса текущего кадра
vco: float                               # legacy поле, не используется
```

### CurrentOutput / WindOutput
Вспомогательные структуры в `structs.py` (не упаковываются в UDP напрямую):
```python
CurrentOutput(u_x, u_y)   # [м/с], географический фрейм; .speed, .direction — derived properties
WindOutput(direction, sig) # wdir [°] и средняя интенсивность бэкскаттера
```

### Processor.update() return
`update(back, navi)` возвращает dict — Manager конвертирует его в `ProcessResult`:
```python
{"out": Output | None, "pulse": int, "step": float, "navi": Navi, "port": ndarray | None}
```
`"out"` = `None` пока `s.index < N_SHOTS` или условие `% out_times` не выполнено.

### ProcessResult
Передаётся из Processor в output sinks через `Manager.out_queue`:
```python
ProcessResult(output: Output, port: np.ndarray, navi: Navi)
```

### OutputSink (интерфейс)
Новые sinks наследуются от `OutputSink` (`src/io/output.py`) и реализуют:
```python
def send(self, result: ProcessResult): ...  # обязательный
def close(self): ...                        # необязательный (вызывается при shutdown)
```

### UDP пакет (2420 байт)
```
"<BBHHBBHHHHHBBHHHHHHHHHHHHHhHH{N_FREQ}B{N_FREQ×N_DIRS}B"
 ────────────────────────────────────────────────────────────────────
 52 байт заголовок + N_FREQ байт spec_1d + N_FREQ×N_DIRS байт spec_2d
```
Индексы `un[]` в `tester_receive.py`:
- `un[3]`  = `rps * 100`         (восстановлен; угловая скорость)
- `un[6]`  = `step_area * 1000`  (восстановлен; шаг сегмента)
- `un[7]`  = `curr_dir`          (целые градусы, 1°; был n_area)
- `un[8]`  = `n_start` → **HDG** (целые градусы)
- `un[9]`  = `cog_proc * 100`
- `un[10]` = `sog_proc * 100`
- `un[11]` = `curr_speed * 100`  (uint8, [cm/s]; был max_sys)
- `un[12]` = `ide_sys`
- `un[13..15]` = wave_sum: `swh*100`, `d_p*100`, `t_p*100`
- `un[16..18]` = wave_win (ветровая): swh, d_p, t_p
- `un[19..21]` = wave_sw1
- `un[22..24]` = wave_sw2
- `un[25]` = quality flag (h, signed; 0 = плохо, 1 = хорошо)
- `un[26]` = 0 (зарезервировано; был vco/u_proj)
- `un[27]` = `wind_dir` [°]  (НОВОЕ поле)
- `un[28]` = `wspd × 10` [0.1 м/с]  (НОВОЕ поле)
- `un[29..]` = spec_1d, затем spec_2d

### CSVOutputSink
При `file = true` создаёт 4 файла с timestamp-префиксом `YYYYMMDDTHHMMSS`:
- `_params.csv` — `datetime;pulse;step;swh;t_p;d_p;d_m;t_m;freq[0..N_FREQ-1]`
- `_port.csv` — ω-k портрет `(N_SHOTS//2, K_NUM)`
- `_spec.csv` — направленный спектр `(N_DIRS, N_FREQ)`
- `_navi.csv` — `datetime,lat,lon,spd,sog,cog,hdg`

`save_path` в `[output]` должен заканчиваться разделителем (`/` на Linux, `\` на Windows) — путь строится как `save_path + timestamp + "_port.csv"` без дополнительных разделителей.

## Файлы проекта

| Файл | Назначение |
|------|-----------|
| `src/processing/processor.py` | Главный процессор, `update()` — основной цикл; `_SIGNAL_BAND=10` |
| `src/processing/state.py` | `ProcessorState` — состояние между кадрами |
| `src/processing/averaging.py` | `Averager` — кольцевой буфер, нормировка в [0,255] |
| `src/algorithms/spectrum2d.py` | `calc_spec3d`, `calc_port`, `apply_doppler_3d_vec`, `separate_signal_noise`, `apply_mtf`, `compute_snr`, `compute_frequency_spectrum`, `calc_spec2d` |
| `src/algorithms/dispersion.py` | `calc_current_vector` (МНК по 3D спектру), `calc_vco` (legacy), `dispersion_curve` |
| `src/algorithms/partition.py` | `calc_wspd` (ветер), `calc_partitions` (разбиение на системы) |
| `src/algorithms/area.py` | `Area.calc_mask()` — вырезка сегмента с билинейной интерполяцией |
| `src/config.py` | `load_config()` → `AppConfig(Constants, PipelineConfig, input, output)` |
| `src/io/structs.py` | `Wave`, `WaveOutput`, `Output`, `ProcessResult`, `BackData`, `BackPack`, `Navi`; `parse_back_packet()`, `parse_navi_packet()` → поднимают `ProtocolError` при неверном формате пакета |
| `src/io/output.py` | `OutputSink` (база), `UdpOutputSink`, `CSVOutputSink`; `_wlen(t_p)` — λ из периода |
| `src/io/input.py` | `UdpInputSource`, `NCInputSource`, `BT8InputSource` |
| `src/io/service.py` | Создание UDP-сокетов |
| `src/runtime/manager.py` | Трёхпоточный пайплайн с watchdog |
| `src/runtime/logger.py` | `setup_logger()` |
| `config.ini` | Все константы и настройки |
| `test/tester_receive.py` | Визуализация: 1D спектр + полярный спектр + таблица параметров |
| `test/tester_transmit.py` | Генератор тестового UDP-потока (без радара) |
| `src/processing/processor (copy).py` | Референсная копия processor.py (не импортируется) |
| `batch_process.py` | Автономная пакетная обработка NC-файлов; дублирует алгоритмы processor.py без усреднения; `_SIGNAL_BAND=10` продублирован — менять синхронно с `processor.py`; для файлов с `"0606"` в имени опционально загружает данные буя (`ewdm`, `xarray`) и рисует EWDM-спектр; фигура всегда 3 столбца — [0,2] EWDM или пусто, [1,2] ω-k портрет (до Doppler-коррекции) с невозмущённой кривой и идентифицированным сдвигом; `wind_meta={'u_10','v_10'}` из метаданных добавляет ERA5-направление на полярный спектр |
| `batch_process_parallel.py` | Параллельная обёртка над `batch_process._process_file`: локальный `multiprocessing.Pool` (`--n-workers`) или SLURM-array (`--task-id`/`--n-tasks`, страйдовая нарезка); `--merge-only` сливает partial CSV из `{out}/partial/` |
| `run_batch.sh` | SLURM sbatch-скрипт: партиция `r2c2`, 48 CPU, 160 GB, запускает `batch_process_parallel.py --n-workers 48` |

### Устаревшие файлы
`src/algorithms/portrait.py`, `src/algorithms/direction.py` — старые реализации, не импортируются из `processor.py`. Не трогать и не реанимировать.

## Конфигурация (config.ini)

Секции: `[constants]` — физические параметры; `[pipeline]` — `queue_size`, `restart_on_error`; `[input]` — `type`, `my_ip`, `back_port`, `navi_port`, `data_path` и др.; `[output]` — `udp`, `file`, `server_ip`, `server_port`, `save_path`, `out_times`, `pics`.

Обычно стабильные значения:
```
AAP=4096  ARDP=2048  ADP=1192  ASP=192
K_NUM=32  N_SHOTS=256  N_FREQ=64  N_DIRS=36
NUM_AREA=8  RPM=25  MEAN=4
CHANGE_DIR_NUM_SHOTS=16  ← загружается в Constants, но Processor не использует (legacy)
```
`SNR_A`, `SNR_B`, `WSPD_A`, `WSPD_B` — меняются при рекалибровке; **всегда смотреть в config.ini**. Коэффициенты WSPD в config.ini умноженные на 0.01 в коде дают м/с.

Флаги качества (захардкожены, **не в config.ini**; значения в `processor.py` и `batch_process.py` **различаются**):

| Параметр | `processor.py` | `batch_process.py` |
|---|---|---|
| `_SNR_QUALITY_MIN` | 1.5 | 1.5 |
| `_WIND_SIG_MIN` | 10.0 | 5.0 |
| `_T_PEAK_MIN` | 6.0 | 5.5 |

- `quality = 1` если все три условия выполнены И `n_sys >= 1`
- `ring_sig` = `std` интенсивности бэкскаттера в кольце `ADP±ASP` (≠ `sig` из `calc_wspd`, которая — среднее по всему изображению)
- В `processor.py`: `result.n_dis = quality` → `un[25]` в UDP пакете (0 = плохо, 1 = хорошо)
- При `quality=0`: причина выводится в лог (`_log.info` / `log.info`)

`om_max = π · RPM / 60 ≈ 1.309 rad/s` (частота Найквиста).

Имена в ini → имена в коде: `AREA_AZIM_PX=AAP`, `AREA_READ_DIST_PX=ARDP`, `AREA_DISTANCE_PX=ADP`, `AREA_SIZE_PX=ASP`.

## Debug-выводы

`Processor(config, pics)` — `pics=False` или строка `"false"` отключают отладку; строка с путём к директории (или `"."`) включают. В `main.py` передаётся из `cfg.output.get("pics", "false")`.

`pics` в `[output]`: `false` — отключено; путь к директории (или `"."`) — PNG рядом с `main.py`:
- `debug_portrait.png` — Doppler-скорректированный ω-k портрет + кривая `ω=√(gk)` + per-system residual curves; в заголовке `Ux`, `Uy`
- `debug_spec2d.png` — направленный спектр `s_om_th` + маркеры систем (sum, w_s, sw_1, sw_2)
- `debug_segments.png` — сетка NUM_AREA строк × **3 столбца**: средний бэкскаттер | сырой ω-k портрет | kx-ky срез (сумма по ω > 0.1·om_max)

## Визуализация (tester_receive.py)

- `F_DISPLAY = 0.20` Hz — радиальный предел полярного спектра
- Colormap: белый (0) → тёмно-синий → зелёный → жёлтый → красный
- `_RENDER_DT = 0.35 с` — декаплинг приёма UDP от отрисовки

## Что НЕ нужно делать

- **Не применять `apply_doppler_2d` и скалярный `apply_doppler_3d`** — оба устарели. Использовать только `apply_doppler_3d_vec(spec_3d, k_max, Ux, Uy, om_max)`. `apply_doppler_2d` работает только с 2D портретом и игнорирует направление волны.
- **Не прибавлять скорость судна к `(Ux, Uy)` перед Doppler-коррекцией** — `(Ux, Uy)` для `apply_doppler_3d_vec` правильно включает скорость судна. Прибавлять SOG нужно только при вычислении `u_proj` для вывода.
- **Не использовать SOG в узлах** — `navi.sog` и `navi.spd` уже в м/с.
- **Не добавлять длины волн в пакет** — длины волн удалены из UDP пакета и нигде не вычисляются.
- **Не использовать** удалённые поля Wave: `per`, `len`, `dir`, `ddir`, `vco`, `inv`.
- **Не менять бинарный формат UDP пакета** — `tester_receive.py` зависит от него.
- **Не передавать `--config` в командной строке** — путь захардкожен в `main.py`.

## Ссылки на литературу

Методология Доплер-коррекции: Carrasco, Lund, Nieto-Borge, Young.
Принцип: векторная поправка `Δω = kx·Ux + ky·Uy` корректна для любого числа волновых систем одновременно.
