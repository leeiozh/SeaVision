# SeaVision

Система обработки данных морского навигационного радара для оценки параметров волнения в реальном времени.

---

## Что делает система

Принимает сырой бэкскаттер (UDP или файл) + навигацию, вычисляет:

- Значительную высоту волн Hs, пиковый период Tp, среднее направление
- Декомпозицию на системы: ветровые волны + до 2 зыбей
- Вектор поверхностного течения (Восток, Север) [м/с]
- Направление и скорость ветра по бэкскаттеру
- Направленный спектр S(θ, f) и 1D спектр S(f)

Все результаты выдаются по UDP (2420 байт) и/или в CSV. Протокол: `udp_protocol_ru.md`.

---

## Быстрый старт

```bash
# Активация окружения (Python 3.12)
source sv_env/bin/activate

# Запуск процессора (конфиг: config.ini, захардкожен в main.py)
python main.py

# Визуализация выхода (отдельный терминал, UDP порт 4000)
python test/tester_receive.py

# Тестовый поток без радара
python test/tester_transmit.py
```

---

## Источники входных данных

| `type=` в config.ini | Описание |
|---|---|
| `udp` | Живой радар. Пакеты 1032 байт, сборка по AAP=4096 строк в кадр |
| `nc` | NetCDF-файл с историческими данными (`data_path`) |
| `bt8` | Папка с бинарными BT8-файлами (`bt8_folder`) |

---

## Архитектура

```
UdpInputSource / NCInputSource / BT8InputSource
    ↓  BackData + Navi
Manager  (3 потока: Input → Process → Output)
    ↓
Processor.update()
    ├─ 8 пространственных сегментов → 3D Welch FFT → spec_3d
    ├─ Pre-analysis: ship-Doppler коррекция → find_freq_peaks → find_system_dirs
    ├─ Оценка тока: calc_current_multiwave (≥2 систем) / calc_current_vector
    ├─ Финальная векторная Doppler-коррекция → spec_3d_fixed
    ├─ MTF, SNR, Tp, Tm, peak_dir
    ├─ calc_partitions → ветровые волны / зыбь 1 / зыбь 2
    └─ quality flag
    ↓
Averager (кольцевой буфер MEAN=4, нормировка спектров в [0,255])
    ↓
UdpOutputSink (2420 байт)  /  CSVOutputSink (4 файла)
    ↓
test/tester_receive.py  (визуализация)
```

**Триггер расчёта**: `index >= N_SHOTS` и `index % out_times == 0`.

---

## Пакетная обработка

```bash
# Последовательная — один процесс
python batch_process.py [--csv META_upd2.csv] [--base-path /storage/...] [--out batch_out] [--config config.ini]

# Параллельная — локальный multiprocessing
python batch_process_parallel.py --n-workers 8

# Параллельная — SLURM array
python batch_process_parallel.py --task-id $SLURM_ARRAY_TASK_ID --n-tasks $SLURM_ARRAY_TASK_COUNT

# Слияние partial CSV после SLURM
python batch_process_parallel.py --merge-only --out batch_out
```

Вывод: `{out}/params.csv` + `{out}/spec/{name}_freqspec.npy` / `_dirspec.npy` + `{out}/pics/{name}.png`

---

## UDP-протокол выхода

Полное описание — `udp_protocol_ru.md`. Краткая шапка (52 байт + 64 + 2304):

| Поле | Индекс un[] | Кодирование | Конвенция |
|---|---|---|---|
| Тип пакета | 0 | всегда = 5 | — |
| Импульс (SP/MP/LP) | 1 | 1/2/3 | — |
| Range resolution | 2 | мм (×1000) | — |
| RPM | 3 | об/мин × 100 | — |
| curr_dir | 7 | градусы целые | **КОМПАС** |
| HDG | 8 | градусы целые | — |
| curr_speed | 11 | (м/с)×100, uint8 | — |
| n_sys | 12 | 0–3 | — |
| swh/dir/t_p суммарные | 13–15 | ×100 | dir: **МАТ.** |
| ветер / зыбь 1 / зыбь 2 | 16–24 | ×100 | dir: **МАТ.** |
| quality | 25 | 0=BAD 1=GOOD | signed int16 |
| wind_dir | 27 | градусы целые | **МАТ.** |
| wspd | 28 | (м/с)×10 | — |
| spec_1d | 29..92 | uint8 [0..255] | — |
| spec_2d | 93.. | uint8 [0..255], 36×64 | строка i = i×10° МАТ. |

**МАТ.** = математическая конвенция: 0°=Восток, 90°=Север, против часовой.  
Перевод в компас: `(90 − θ_мат) mod 360`.

---

## Конфигурация (config.ini)

Ключевые константы:

| Параметр | Значение | Описание |
|---|---|---|
| AAP | 4096 | Азимутальных линий на оборот |
| ARDP | 2048 | Пикселей по дальности |
| ADP | 1192 | Расстояние до центра сегмента [px] |
| ASP | 192 | Полуразмер сегмента [px] |
| K_NUM | 32 | Пространственных частотных бинов |
| N_SHOTS | 256 | Кадров в расчётном окне |
| N_FREQ | 64 | Частотных бинов в спектре выхода |
| N_DIRS | 36 | Угловых бинов в направленном спектре |
| NUM_AREA | 8 | Пространственных сегментов |
| RPM | 25 | Скорость вращения антенны [об/мин] |
| MEAN | 4 | Глубина скользящего усреднения |
| SNR_A, SNR_B | *config* | Калибровка Hs: H=0.01×(A+B×√SNR) |
| WSPD_A, WSPD_B | *config* | Калибровка ветра: V=0.01×(A+B×σ_ring) |

---

## Структура файлов

```
main.py                         — точка входа (конфиг захардкожен как config.ini)
config.ini                      — все настройки
src/
  processing/
    processor.py                — Processor.update(): главный цикл вычислений
    averaging.py                — Averager: кольцевой буфер, нормировка спектров
    state.py                    — ProcessorState
  algorithms/
    spectrum2d.py               — calc_spec3d, apply_doppler_3d_vec, calc_spec2d, ...
    dispersion.py               — calc_current_vector, calc_current_multiwave
    partition.py                — find_freq_peaks, find_system_dirs, calc_partitions, calc_wspd
    area.py                     — Area: вырезка сегмента с билинейной интерполяцией
  io/
    input.py                    — UdpInputSource, NCInputSource, BT8InputSource
    output.py                   — UdpOutputSink, CSVOutputSink
    structs.py                  — Wave, Output, ProcessResult, BackData, Navi, ...
    service.py                  — UDP-сокеты
  runtime/
    manager.py                  — 3-поточный пайплайн с watchdog
  config.py                     — load_config() → AppConfig
test/
  tester_receive.py             — визуализация UDP-выхода (канонический)
  tester_transmit.py            — генератор тестового потока (канонический)
batch_process.py                — пакетная обработка NC-файлов
batch_process_parallel.py       — параллельная обёртка (multiprocessing / SLURM)
run_batch.sh                    — SLURM sbatch-скрипт (r2c2, 48 CPU)
debug_multiwave.py              — диагностика мультиволнового пайплайна на одном NC
compare_pipelines.py            — сравнение batch vs processor (диагностика)
udp_protocol_ru.md              — полный протокол UDP-пакета (v1.1)
```

---

## Сборка в исполняемый файл

```bash
pyinstaller main.py --onedir
```

---

## Флаги качества

`quality = 1` (GOOD) при одновременном выполнении всех условий:

| Критерий | Порог |
|---|---|
| SNR (сигнал/шум в ω-k) | ≥ 5.0 |
| σ_ring (std бэкскаттера в кольце ADP±ASP) | ≥ 5.5 |
| T_peak | ≥ 5.5 с |
| n_sys | ≥ 1 |
