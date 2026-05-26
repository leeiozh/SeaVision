#!/usr/bin/env python3
"""
Parallel wrapper around batch_process.

Modes
-----
Local multiprocessing:
    python batch_process_parallel.py --n-workers 8

SLURM array (called from run_batch.sh — do not invoke manually):
    python batch_process_parallel.py --task-id $SLURM_ARRAY_TASK_ID \
                                     --n-tasks  $SLURM_ARRAY_TASK_COUNT

Merge partial CSVs after array job completes:
    python batch_process_parallel.py --merge-only --out batch_out
"""

import argparse
import gc
import os

import numpy as np
import pandas as pd
from multiprocessing import Pool

from src.config import load_config
from src.runtime.logger import setup_logger
from batch_process import _process_file, _PARAMS_FIELDS


# ── worker (top-level for pickle) ────────────────────────────────────────────

def _worker(packed):
    name, nc_path, cfg, spec_dir, pics_dir, wind_meta = packed
    log = setup_logger(f'bp.{os.getpid()}')
    result = _process_file(name, nc_path, cfg, spec_dir, pics_dir, log,
                           wind_meta=wind_meta)
    return name, result


# ── core loop ─────────────────────────────────────────────────────────────────

def _run(file_pairs, cfg, spec_dir, pics_dir, log, n_workers, out_csv):
    # Resume support: skip files already recorded in out_csv
    done = set()
    if os.path.exists(out_csv):
        try:
            done = set(pd.read_csv(out_csv, usecols=['name'])['name'].dropna())
            if done:
                log.info(f'Resume: skipping {len(done)} already-done files')
        except Exception as exc:
            log.warning(f'Could not read existing {out_csv}: {exc}')

    pending = [(n, p, wm) for n, p, wm in file_pairs if n not in done]
    if not pending:
        log.info('All files already processed')
        return

    tasks = [(name, path, cfg, spec_dir, pics_dir, wind_meta)
             for name, path, wind_meta in pending]

    # CSV is written from the main process after each result — no lock needed
    write_header = not os.path.exists(out_csv)
    n_done = 0

    def _handle(name, res):
        nonlocal write_header, n_done
        if res is None:
            return
        row, _s1, _s2 = res
        pd.DataFrame([row]).reindex(columns=_PARAMS_FIELDS).to_csv(
            out_csv, mode='a', header=write_header, index=False, float_format='%.4f'
        )
        write_header = False
        n_done += 1
        log.info(f'[{n_done}/{len(pending)}] {name}: '
                 f'quality={row["quality"]}  swh={row["swh"]:.2f}m')

    if n_workers > 1:
        with Pool(n_workers) as pool:
            for name, res in pool.imap_unordered(_worker, tasks):
                _handle(name, res)
                gc.collect()
    else:
        for t in tasks:
            name, res = _worker(t)
            _handle(name, res)
            gc.collect()

    log.info(f'Done: {n_done} new rows → {out_csv}')


# ── merge ─────────────────────────────────────────────────────────────────────

def _merge(out_dir):
    partial_dir = os.path.join(out_dir, 'partial')
    files = sorted(
        f for f in os.listdir(partial_dir)
        if f.startswith('params_') and f.endswith('.csv')
    )
    if not files:
        print(f'No partial CSVs found in {partial_dir}')
        return
    merged = pd.concat(
        [pd.read_csv(os.path.join(partial_dir, f)) for f in files],
        ignore_index=True,
    )
    out_path = os.path.join(out_dir, 'params.csv')
    merged.to_csv(out_path, index=False)
    print(f'Merged {len(files)} files → {len(merged)} rows → {out_path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Parallel SeaVision batch processing')
    parser.add_argument('--csv',        default='META_upd.csv')
    parser.add_argument('--base-path',  default='/storage/thalassa/DATA/RADAR/')
    parser.add_argument('--out',        default='batch_out')
    parser.add_argument('--config',     default='config.ini')
    parser.add_argument('--n-workers',  type=int, default=1,
                        help='Worker processes (local mode)')
    parser.add_argument('--task-id',    type=int, default=None,
                        help='SLURM_ARRAY_TASK_ID (0-based, set by run_batch.sh)')
    parser.add_argument('--n-tasks',    type=int, default=None,
                        help='SLURM_ARRAY_TASK_COUNT (set by run_batch.sh)')
    parser.add_argument('--merge-only', action='store_true',
                        help='Only merge partial CSVs from a previous array run')
    args = parser.parse_args()

    if args.merge_only:
        _merge(args.out)
        return

    log = setup_logger('batch_par')
    cfg = load_config(args.config)

    spec_dir    = os.path.join(args.out, 'spec')
    pics_dir    = os.path.join(args.out, 'pics')
    partial_dir = os.path.join(args.out, 'partial')
    for d in (args.out, spec_dir, pics_dir):
        os.makedirs(d, exist_ok=True)

    df_all = pd.read_csv(args.csv)
    df_all = df_all[df_all["cruise"] == "ai63"]
    df_all = df_all[df_all["num"] > 430]

    slurm_mode = args.task_id is not None and args.n_tasks is not None
    if slurm_mode:
        # Strided slice: task 0 → rows 0, N, 2N, …; task 1 → 1, N+1, …
        os.makedirs(partial_dir, exist_ok=True)
        indices = list(range(args.task_id, len(df_all), args.n_tasks))
        if not indices:
            log.info(f'Task {args.task_id}: nothing to process (n_files={len(df_all)})')
            return
        out_csv = os.path.join(partial_dir, f'params_{args.task_id:04d}.csv')
        log.info(f'SLURM task {args.task_id}/{args.n_tasks}: '
                 f'{len(indices)} files  → {out_csv}')
    else:
        indices = list(range(len(df_all)))
        out_csv = os.path.join(args.out, 'params.csv')
        log.info(f'Local mode: {len(indices)} files, {args.n_workers} workers')

    has_wind = {'u_10', 'v_10'}.issubset(df_all.columns)
    file_pairs = []
    for idx in indices:
        row     = df_all.iloc[idx]
        name    = row['name'].split('/')[-1][:-3]
        nc_path = os.path.join(args.base_path, row['name'])
        wind_meta = None
        if has_wind and pd.notna(row.get('u_10')) and pd.notna(row.get('v_10')):
            wind_meta = {'u_10': float(row['u_10']), 'v_10': float(row['v_10'])}
        file_pairs.append((name, nc_path, wind_meta))

    _run(file_pairs, cfg, spec_dir, pics_dir, log, args.n_workers, out_csv)


if __name__ == '__main__':
    main()
