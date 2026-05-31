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
python batch_process.py [--csv META_upd2.csv] [--base-path /storage/thalassa/DATA/RADAR/] [--out batch_out] [--config config.ini]
# CSV: любые столбцы + 'name' с путями к .nc; --base-path — префикс к каждому пути
# вывод: {out}/params.csv + {out}/spec/{name}_freqspec.npy / _dirspec.npy + {out}/pics/{name}.png

# Параллельная пакетная обработка (multiprocessing или SLURM)
python batch_process_parallel.py --n-workers 8   # локальный режим
python batch_process_parallel.py --task-id $SLURM_ARRAY_TASK_ID --n-tasks $SLURM_ARRAY_TASK_COUNT  # SLURM
python batch_process_parallel.py --merge-only --out batch_out   # слияние partial CSV после SLURM
# run_batch.sh — SLURM sbatch-скрипт (48 воркеров, партиция r2c2)

# Диагностика мультиволнового пайплайна на одном NC-файле
python debug_multiwave.py   # путь NC захардкожен внутри, вывод → ./debug_out/

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

**COG в processor.py** — вычисляется как circular mean (через arctan2 от mean(sin), mean(cos)) по кольцевому буферу `s.cog` размера MEAN, чтобы избежать артефактов на границе 0°/360°. Аналогично batch_process.py.

## Ключевые алгоритмы

### Методология обработки (актуальная, после рефакторинга мая–июня 2026)

**Шаги 1–2: накопление спектра**

1. **Все NUM_AREA сегментов одновременно, без поворота** (`orient=0`).
   Каждый сегмент — квадрат `2·ASP × 2·ASP` пикселей, центр на расстоянии `ADP` px под своим азимутом.
   Азимуты: `np.linspace(0, 360, NUM_AREA, endpoint=False)`.

2. **3D Welch FFT каждого сегмента** → `spec_3d_i (N_SHOTS//2, 2·K_NUM, 2·K_NUM)`, усреднение по NUM_AREA → `spec_3d_corr`.
   Механика `calc_spec3d`: пространственный `fft2` по осям (1,2) для каждого кадра → `welch` по временной оси (0) → берётся положительная половина (N_SHOTS//2 бинов).

**Шаги 3–4: идентификация волновых систем и оценка тока**

3. **Pre-analysis на ship-corrected спектре:**
   - `apply_doppler_3d_vec(spec_3d_corr, Ux_ship, Uy_ship)` → `spec_3d_ship`
     где `Ux_ship = SOG·sin(COG)`, `Uy_ship = SOG·cos(COG)`.
   - `find_freq_peaks(s_omega_pre, omega_vals)` → список пиков `{om, om_lo, om_hi}` из 1D MTF-спектра.
   - `calc_spec2d(spec_3d_ship, ...)` → `s_om_th_pre`.
   - `find_system_dirs(s_om_th_pre, freq_peaks, ...)` → `systems_draft` — список `{om, om_lo, om_hi, dir_deg}`.

4. **Оценка вектора тока `(Ux, Uy)` — кажущаяся скорость в радарном фрейме = ток_воды + v_судна:**
   - Если `len(systems_draft) >= 2`: **`calc_current_multiwave`** — двухпроходный МНК раздельно по
     каждой системе (k-диапазон + угловой конус `±_MULTI_DIR_HALF_DEG=45°`), нормировка веса
     по системе (равный вклад независимо от энергии).
     Pass 1: argmax в широком окне `wide = max(band, n_om//4)` без prior → грубая оценка.
     Pass 2: centroid в узком окне `±band` вокруг `ω_ref + kx·Ucx0 + ky·Ucy0` → точная оценка.
     Широкое окно в Pass 1 позволяет захватить энергию при сильном токе (до ~2 м/с),
     которую узкое окно вокруг `ω_ref` пропустило бы. Возвращает остаточный `(Ucx, Ucy)`;
     `Ux = Ucx + Ux_ship`. При плохой обусловленности (sv_min/sv_max < `_MULTI_MIN_SV_RATIO`)
     возвращает `(None, None)` → fallback.
   - Иначе или при fallback: **`calc_current_vector`** — двухпроходной argmax+centroid по
     всем валидным `(kx, ky)` ячейкам. Pass 1 — argmax в широком окне, инициализация от
     `(Ux_ship, Uy_ship)`; Pass 2 — centroid в узком окне `_SIGNAL_BAND`.

**Шаги 5–11: финальный спектр и разбиение**

5. **Векторная Допплер-коррекция** `apply_doppler_3d_vec(spec_3d_corr, k_max, Ux, Uy, om_max)`:
   каждая ячейка `(kx, ky)` сдвигается на `Δω = kx·Ux + ky·Uy` — точно для любого числа волновых систем одновременно → `spec_3d_fixed`.

6. **ω-k портрет** `port_fixed` из `spec_3d_fixed` через `calc_port`.

7. **Разделение сигнал/шум**: полоса `±_SIGNAL_BAND=10 бинов` вокруг `ω = √(g·k)` — единый параметр для всех шагов.

8. **MTF коррекция**: `k^{-1.2}`.

9. **1D ω-спектр** из MTF-взвешенного сигнала → `T_peak`, `T_mean`, `m0`, `snr_tot`.

10. **Направленный спектр** `s_om_th (N_DIRS, N_SHOTS//2)` из `spec_3d_fixed` через `calc_spec2d` с тем же `band=_SIGNAL_BAND`.
    **`peak_dir` — математическая конвенция**: 0° = Восток (+X сегмента), 90° = Север, 180° = Запад, 270° = Юг.
    Проекция скорости на направление волны: `u_proj = Ux·cos(peak_dir) + Uy·sin(peak_dir)`.
    **Не использовать компасную формулу** `Ux·sin + Uy·cos` — sin/cos перепутаны.

11. **Разбиение на системы** `calc_partitions` — итеративный поиск пиков в `s_om_th` с гашением.
    Жёсткий кэп: `len(systems) >= 3 → break` в начале каждой итерации — гарантирует `n_sys ≤ 3`.
    Классификация: система в пределах `±_PART_WIND_DIR_THRESH=45°` от wdir = ветровая (кратчайший T из кандидатов); нет кандидата → все = зыби.
    Сохранение энергии: raw fracs нормируются → `h_s_i = swh·√(frac_norm_i)`, `Σ h_s_i² = swh²`.
    **Порядок зыбей**: `swell_sys` сортируется по возрастанию T, поэтому `sw_1` = зыбь с наименьшим T (высокочастотная), `sw_2` = более длиннопериодная зыбь.

### Ток (течение)
Координатная система сегмента (`orient=0`): строки (axis 0) идут с Запада на Восток, столбцы (axis 1) — с Юга на Север. После FFT:
- **kx (axis 0) = Восток**, ky (axis 1) = Север

`(Ux, Uy)` из `calc_current_vector` / `calc_current_multiwave` — кажущаяся скорость в координатах **(Восток, Север)**.

Истинный ток воды (вычитаем скорость судна, т.к. `(Ux,Uy) = v_curr + v_ship`):
```
u_curr_East  = Ux − SOG · sin(COG)   # Восток
u_curr_North = Uy − SOG · cos(COG)   # Север
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
Возвращает `(sig, wdir_deg)`, где `sig = mean(bck)` — среднее значение бэкскаттера всего изображения (≠ `ring_sig`). `wdir` передаётся в `calc_partitions()` только как ориентир классификации (система в пределах ±45° от wdir = ветровая); `wave_win.d_p` — спектральный пик ветровой системы, а не `wdir` напрямую.

### Averager
Хранит кольцевой буфер последних `MEAN` выходов. `get_mean()` усредняет их (до `min(index, MEAN)` накоплений) и **нормирует спектральные массивы в [0, 255]** перед возвратом. Возвращает `(Output, spec_1d, spec_2d, port)` — все spectral arrays нормированы в `[0, 255]` и приведены к `int`; поля Wave (swh, t_p и т.д.) нормировке не подвергаются.

В `processor.update()` возвращённые `spec_1d`/`spec_2d` (позиции 2,3) не используются; `port` (позиция 4) идёт в `ProcessResult.port` → `_port.csv` sink.

## Основные структуры

### Wave
```python
Wave(swh, snr, t_p, t_m, d_p, d_m)
# t_p = пиковый период [с]
# t_m = средний период [с]
# d_p = пиковое направление [°], математическая конвенция
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
    n_dis,                     # quality flag (0=плохо, 1=хорошо)
    spec_1d,                   # нормированный 1D спектр [0..255], int array
    spec_2d,                   # нормированный 2D спектр [0..255], int array
)
# Дополнительные поля, устанавливаемые в processor.update() после get_mean():
# .curr_speed — модуль истинного тока воды [м/с]
# .curr_dir   — компасный курс тока [°]
# .wind_dir   — направление ветра по бэкскаттеру [°]
# .wspd       — скорость ветра [м/с] = 0.01·(WSPD_A + WSPD_B·ring_sig)
```

### ProcessorState
```python
index: int                                # монотонный счётчик кадров; триггер расчёта: index >= N_SHOTS and index % out_times == 0
cbck: (NUM_AREA, N_SHOTS, 2·ASP, 2·ASP)  # float32, 4D буфер всех сегментов
speed, heading, cog: (MEAN,)              # скользящие окна нави-данных [м/с и °]
curr_step: float                          # range resolution [м/px] текущего кадра
curr_pulse: int                           # код импульса текущего кадра
vco: float                               # legacy поле, не используется
indices: np.ndarray                       # legacy: инициализируется arange(N_SHOTS), roll-ится каждый кадр, но никогда не читается
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
- `un[3]`  = `rps * 100`         (угловая скорость)
- `un[6]`  = `step_area * 1000`  (шаг сегмента)
- `un[7]`  = `curr_dir`          (целые градусы; был n_area)
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
- `un[26]` = 0 (зарезервировано)
- `un[27]` = `wind_dir` [°]
- `un[28]` = `wspd × 10` [0.1 м/с]
- `un[29..]` = spec_1d, затем spec_2d

### CSVOutputSink
При `file = true` создаёт 4 файла с timestamp-префиксом `YYYYMMDDTHHMMSS`:
- `_params.csv` — `datetime;pulse;step;swh;t_p;d_p;d_m;t_m;freq[0..N_FREQ-1]`
- `_port.csv` — ω-k портрет `(N_SHOTS//2, K_NUM)`
- `_spec.csv` — направленный спектр `(N_DIRS, N_FREQ)`
- `_navi.csv` — `datetime,lat,lon,spd,sog,cog,hdg`

`save_path` в `[output]` должен заканчиваться разделителем (`/` на Linux, `\` на Windows).

## Файлы проекта

| Файл | Назначение |
|------|-----------|
| `src/processing/processor.py` | Главный процессор, `update()` — основной цикл; `_SIGNAL_BAND=10`; debug-функции `_save_debug_portrait`, `_save_debug_freq_spec`, `_save_debug_spec2d`, `_save_debug_segments` |
| `src/processing/state.py` | `ProcessorState` — состояние между кадрами |
| `src/processing/averaging.py` | `Averager` — кольцевой буфер, нормировка в [0,255] |
| `src/algorithms/spectrum2d.py` | `calc_spec3d`, `calc_port`, `apply_doppler_3d_vec`, `separate_signal_noise`, `apply_mtf`, `compute_snr`, `compute_frequency_spectrum`, `calc_spec2d` |
| `src/algorithms/dispersion.py` | `calc_current_vector` (двухпроходной МНК), `calc_current_multiwave` (per-system МНК при ≥2 системах, с `sys_scatter`), `calc_vco` (legacy), `dispersion_curve`; пороги `_MULTI_*` |
| `src/algorithms/partition.py` | `calc_wspd`, `find_freq_peaks`, `find_system_dirs`, `calc_partitions`; пороги `_FPEAK_*`, `_SDIR_*`, `_PART_*` |
| `src/algorithms/area.py` | `Area.calc_mask()` — вырезка сегмента с билинейной интерполяцией |
| `src/config.py` | `load_config()` → `AppConfig(Constants, PipelineConfig, input, output)` |
| `src/io/structs.py` | `Wave`, `WaveOutput`, `Output`, `ProcessResult`, `BackData`, `BackPack`, `Navi`; `parse_back_packet()`, `parse_navi_packet()` → `ProtocolError` при неверном формате |
| `src/io/output.py` | `OutputSink` (база), `UdpOutputSink`, `CSVOutputSink` |
| `src/io/input.py` | `UdpInputSource`, `NCInputSource`, `BT8InputSource` |
| `src/io/service.py` | Создание UDP-сокетов |
| `src/runtime/manager.py` | Трёхпоточный пайплайн с watchdog |
| `src/runtime/logger.py` | `setup_logger()` |
| `config.ini` | Все константы и настройки |
| `test/tester_receive.py` | Визуализация: 1D спектр + полярный спектр + таблица параметров |
| `test/tester_transmit.py` | Генератор тестового UDP-потока (без радара) |
| `batch_process.py` | Пакетная обработка NC-файлов; реализует тот же двухфазный пайплайн (pre-analysis + multiwave/fallback), что и `processor.py`. Диагностическая фигура (3 колонки): [0,0] 1D спектр с цветными полосами систем; [0,1] полярный направленный спектр с крестами `systems_draft`; [0,2] ω-k портрет с цветным scatter по системам; [1,:] кольцо бэкскаттера; [2,:] таблица параметров. Для файлов `"0606"` загружает данные буя (ewdm, xarray). `_SIGNAL_BAND=10` продублирован — менять синхронно с `processor.py`. `wind_meta={'u_10','v_10'}` добавляет ERA5-направление ветра. |
| `batch_process_parallel.py` | Параллельная обёртка над `batch_process._process_file`: локальный `multiprocessing.Pool` (`--n-workers`) или SLURM-array (`--task-id`/`--n-tasks`); `--merge-only` сливает partial CSV из `{out}/partial/` |
| `run_batch.sh` | SLURM sbatch-скрипт: партиция `r2c2`, 48 CPU, 160 GB |
| `debug_multiwave.py` | Автономный диагностический скрипт для одного NC-файла: прогоняет полный мультиволновой пайплайн и сохраняет 4 debug-PNG в `./debug_out/`. Путь NC и параметры захардкожены внутри. |
| `view_res.py` | Черновой просмотр результатов батча из `batch_out/params.csv`. Не синхронизируется. |
| `compare_pipelines.py` | Диагностика расхождений batch_process vs processor на `0606_4338.nc`: тест A — одни кадры, разный ring-buf offset (изолирует Hann-window misalignment); тест B — разные окна кадров (как в реальном processor). Путь NC захардкожен внутри. |

### Устаревшие файлы
`src/algorithms/portrait.py`, `src/algorithms/direction.py` — старые реализации, не импортируются. Не трогать и не реанимировать.

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

Флаги качества (захардкожены, **не в config.ini**; одинаковы в `processor.py` и `batch_process.py`):

| Параметр | Значение |
|---|---|
| `_SNR_QUALITY_MIN` | 5.0 |
| `_WIND_SIG_MIN` | 5.5 |
| `_T_PEAK_MIN` | 5.5 |

- `quality = 1` если все три условия выполнены И `n_sys >= 1`
- `ring_sig` = `std` интенсивности бэкскаттера в кольце `ADP±ASP` (≠ `sig` из `calc_wspd`)
- При `quality=0`: причина выводится в лог

`om_max = π · RPM / 60 ≈ 1.309 rad/s` (частота Найквиста).

Имена в ini → имена в коде: `AREA_AZIM_PX=AAP`, `AREA_READ_DIST_PX=ARDP`, `AREA_DISTANCE_PX=ADP`, `AREA_SIZE_PX=ASP`.

## Алгоритмические пороги (вынесены как module-level константы)

### `src/algorithms/partition.py` — пороги `_FPEAK_*` / `_SDIR_*` / `_PART_*`

| Константа | Значение | Назначение |
|---|---|---|
| `_FPEAK_MIN_PROM_REL` | 0.15 | Минимальная высота пика относительно глобального максимума |
| `_FPEAK_BAND_HALF_FRAC` | 0.10 | Полуширина частотной полосы пика (доля от n_om) |
| `_FPEAK_MIN_SEP_FRAC` | 0.12 | Минимальное расстояние между пиками (доля от n_om) |
| `_FPEAK_SMOOTH_SIGMA` | 2.0 | Гауссово сглаживание перед поиском пиков [бин] |
| `_FPEAK_MAX` | 3 | Максимальное число искомых пиков |
| `_SDIR_SMOOTH_SIGMA` | 1.0 | Сглаживание угловой проекции [бин] |
| `_PART_MIN_DIR_SEP` | 25.0 | Минимальное угловое расстояние между системами [°] |
| `_PART_MIN_PER_RATIO` | 1.1 | Минимальное отношение T_large/T_small для различимости |
| `_PART_MIN_ENERGY_FRAC` | 0.05 | Минимальная доля энергии окна пика в total_energy |
| `_PART_NOISE_SNR` | 3.0 | Пик должен превышать медиану фона × этот коэффициент |
| `_PART_BLANK_FRAC` | 0.08 | Радиус гашения пика [доля от каждой оси] |
| `_PART_WIND_DIR_THRESH` | 45.0 | Максимальный угол от wdir для классификации как ветровая [°] |

### `src/algorithms/dispersion.py` — пороги `_MULTI_*`

| Константа | Значение | Назначение |
|---|---|---|
| `_MULTI_DIR_HALF_DEG` | 45.0 | Полуширина углового конуса вокруг направления системы [°] |
| `_MULTI_MIN_CELLS` | 5 | Минимальное число ячеек для включения системы в МНК |
| `_MULTI_MIN_SV_RATIO` | 0.05 | Минимальное σ_min/σ_max (защита от вырожденной матрицы) |
| `_MULTI_MAX_CURRENT` | 2.55 | Физический клип для остаточного тока [м/с] |
| `_MULTI_K_MIN_REL` | 0.08 | Минимальный k/k_max для включения ячейки (длинная зыбь даёт Δω<0.5 бина) |

## Debug-выводы

`Processor(config, pics)` — `pics=False` или строка `"false"` отключают отладку; строка с путём к директории (или `"."`) включают. В `main.py` передаётся из `cfg.output.get("pics", "false")`.

`pics` в `[output]`: `false` — отключено; путь к директории (или `"."`) — PNG рядом с `main.py`:
- `debug_combined.png` — единый диагностический рисунок (3 панели + таблица параметров):
  - [0,0] 1D частотный спектр `s_omega_pre` (ship-corrected) с полосами систем и Tp/Tm линиями
  - [0,1] полярный направленный спектр `s_om_th` с маркерами систем и COG стрелкой
  - [0,2] ω-k портрет `port_corr` (до коррекции) с кривыми `ship_only` и `full` и scatter из `sys_scatter`
  - [1,:] таблица параметров: Hs, Tp, Dp, системы, Ux/Uy, ток, SNR, WSPD

## Визуализация (tester_receive.py)

- `F_DISPLAY = 0.20` Hz — радиальный предел полярного спектра
- Colormap: белый (0) → тёмно-синий → зелёный → жёлтый → красный
- `_RENDER_DT = 0.35 с` — декаплинг приёма UDP от отрисовки

## UDP-протокол — сводка и авторитетный источник

Полное описание протокола: `udp_protocol_ru.md` (версия 1.1) — **авторитетный источник**.
Старый `udp_protocol.docx` устарел в части именования поля [3]: там оно называлось `rps × 100`, но в коде всегда хранился **RPM × 100** (25 об/мин × 100 = 2500). Новый md это исправляет.

### Конвенции направлений в пакете

| Поле | Конвенция |
|---|---|
| `curr_dir` [7] | **КОМПАС**: 0°=Север, по часовой; `arctan2(East, North) % 360` |
| `dir_sum/win/sw1/sw2` [14,17,20,23] | **МАТ.**: 0°=Восток, против часовой |
| `wind_dir` [27] | **МАТ.**: 0°=Восток, против часовой |
| `spec_2d` строка i | **МАТ.**: направление i×10° (строка 0=Восток, строка 9=Север) |

Перевод математической конвенции в компасную: `(90 − θ_мат) mod 360`.

### Известный визуальный баг: tester_receive.py — полярный спектр

**Проблема**: полярная ось задана в компасной конвенции (`theta_zero='N'`, по часовой), но математические углы из пакета отображаются напрямую без конвертации. Результат: все стрелки направлений и весь спектр повёрнуты на 90° и отражены.

**Не трогать** (баг визуализатора, не протокола/кода):
- В `_update()`: `angle = np.radians(d.get(key, 0))` должно быть `np.radians((90.0 - d.get(key, 0)) % 360)`.
- Theta-edges для `pcolormesh` spec_2d аналогично требуют сдвига.

Это не затрагивает корректность передаваемых данных — только их отображение в `tester_receive.py`.

## Что НЕ нужно делать

- **Не применять `apply_doppler_2d` и скалярный `apply_doppler_3d`** — оба устарели. Использовать только `apply_doppler_3d_vec(spec_3d, k_max, Ux, Uy, om_max)`.
- **Не прибавлять скорость судна к `(Ux, Uy)` перед Doppler-коррекцией** — `(Ux, Uy)` для `apply_doppler_3d_vec` уже включает скорость судна. SOG прибавляется только при вычислении истинного тока для вывода.
- **При multiwave**: `calc_current_multiwave` принимает `spec_3d_ship` (уже скорректированный на судно), возвращает **остаточный** `(Ucx, Ucy)`. Финальный: `Ux = Ucx + Ux_ship`. Не передавать `spec_3d_corr` напрямую в multiwave.
- **Не использовать SOG в узлах** — `navi.sog` и `navi.spd` уже в м/с.
- **Не добавлять длины волн в пакет** — длины волн удалены из UDP пакета и нигде не вычисляются.
- **Не использовать** удалённые поля Wave: `per`, `len`, `dir`, `ddir`, `vco`, `inv`.
- **Не менять бинарный формат UDP пакета** — `tester_receive.py` зависит от него.
- **Не передавать `--config` в командной строке** — путь захардкожен в `main.py`.
- **Не реанимировать** `src/algorithms/portrait.py`, `src/algorithms/direction.py`.
- **При изменении `_SIGNAL_BAND`** — перекалибровать SNR_A/B и изменить синхронно в `processor.py` и `batch_process.py`.
- **Не ориентироваться на старый `udp_protocol.docx`** — использовать `udp_protocol_ru.md`.

## Ссылки на литературу

Методология Допплер-коррекции: Carrasco, Lund, Nieto-Borge, Young.
Принцип: векторная поправка `Δω = kx·Ux + ky·Uy` корректна для любого числа волновых систем одновременно.

Мультиволновая оценка тока: per-system centroid МНК с нормировкой весов (аналог NSP, Dankert & Rosenthal 2004; Stewart & Joy 1974 — концепция эффективной глубины). Работает лучше при угловом разделении систем > 45°; при двух коллинеарных системах матрица A вырождается — автоматический fallback на `calc_current_vector`.
