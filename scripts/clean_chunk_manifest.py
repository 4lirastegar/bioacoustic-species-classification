#!/usr/bin/env python3
"""Remove manifest rows whose chunk WAVs are missing or unreadable (fast, parallel)."""
from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/cache/baseline_mel_chunks/manifest.csv"
BACKUP = MANIFEST.with_suffix(".csv.bak")
DROPPED_LOG = MANIFEST.parent / "manifest_dropped.csv"
WORKERS = 16
MIN_FRAMES = 1000  # ~23 ms at 44.1 kHz — chunks should be ~220500


def check_path(path_str: str) -> tuple[str, bool, str]:
    p = Path(path_str)
    if not p.exists():
        return path_str, False, "missing"
    try:
        if p.stat().st_size < 44:
            return path_str, False, "empty"
        info = sf.info(p)
        if info.frames < MIN_FRAMES:
            return path_str, False, f"short_{info.frames}"
        return path_str, True, "ok"
    except Exception as e:
        return path_str, False, type(e).__name__


def main():
    if not MANIFEST.exists():
        raise SystemExit(f"No manifest at {MANIFEST}")

    m = pd.read_csv(MANIFEST)
    paths = m["path_name"].astype(str).tolist()
    print(f"Checking {len(paths)} paths with {WORKERS} workers...")

    results = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(check_path, p): p for p in paths}
        for fut in tqdm(as_completed(futs), total=len(futs)):
            path_str = futs[fut]
            try:
                path_str, ok, reason = fut.result(timeout=8)
            except Exception:
                ok, reason = False, "timeout"
            results[path_str] = (ok, reason)

    ok_paths = {p for p, (ok, _) in results.items() if ok}
    clean = m[m["path_name"].astype(str).isin(ok_paths)].reset_index(drop=True)
    dropped_n = len(m) - len(clean)
    print(f"OK: {len(clean)} | dropped: {dropped_n}")

    if dropped_n:
        dropped_rows = []
        for _, row in m.iterrows():
            p = str(row["path_name"])
            if p not in ok_paths:
                ok, reason = results[p]
                dropped_rows.append({**row.to_dict(), "reason": reason})
        pd.DataFrame(dropped_rows).to_csv(DROPPED_LOG, index=False)
        print(f"Dropped log -> {DROPPED_LOG}")

    if BACKUP.exists():
        BACKUP.unlink()
    shutil.copy2(MANIFEST, BACKUP)
    clean.to_csv(MANIFEST, index=False)
    print(f"Backup -> {BACKUP}")
    print(f"Clean manifest -> {MANIFEST}")
    print(clean.groupby("subset").size())


if __name__ == "__main__":
    main()
