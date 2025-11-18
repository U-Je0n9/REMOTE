#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remote_re.jsonl → train/val/test 분할 스크립트
- 입력: JSONL (한 줄당 하나의 샘플)
- 출력: train_set_xywh.json / val_set_xywh.json / test_set_xywh.json (기본은 JSON 배열)
"""

import argparse, json, os, random
from collections import defaultdict
from typing import List, Dict, Any, Tuple

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def split_ratios(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    tr, va, te = ratios
    assert abs(tr + va + te - 1.0) < 1e-6, "ratios must sum to 1.0"
    n_tr = int(round(n * tr))
    n_va = int(round(n * va))
    n_te = n - n_tr - n_va
    # rounding 보정
    if n_te < 0:
        # 가장 큰 몫에서 1씩 빼며 보정
        while n_te < 0:
            if n_tr >= n_va:
                n_tr -= 1
            else:
                n_va -= 1
            n_te = n - n_tr - n_va
    return n_tr, n_va, n_te

def stratified_split(items: List[Dict[str, Any]], ratios, seed=42, key="relation"):
    buckets = defaultdict(list)
    for it in items:
        buckets[it.get(key, "__NONE__")].append(it)

    tr, va, te = [], [], []
    rng = random.Random(seed)
    for label, group in buckets.items():
        rng.shuffle(group)
        n_tr, n_va, n_te = split_ratios(len(group), ratios)
        tr.extend(group[:n_tr])
        va.extend(group[n_tr:n_tr+n_va])
        te.extend(group[n_tr+n_va:])
    return tr, va, te

def random_split(items: List[Dict[str, Any]], ratios, seed=42):
    rng = random.Random(seed)
    items = items[:]  # copy
    rng.shuffle(items)
    n_tr, n_va, n_te = split_ratios(len(items), ratios)
    tr = items[:n_tr]
    va = items[n_tr:n_tr+n_va]
    te = items[n_tr+n_va:]
    return tr, va, te

def write_json(path: str, data: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def write_jsonl(path: str, data: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", default="/home/ujeong/KETI/REMOTE/datasets/remote_re.jsonl",
                    help="입력 JSONL 경로")
    ap.add_argument("--outdir", "-o", default="/home/ujeong/KETI/REMOTE/datasets",
                    help="출력 디렉터리")
    ap.add_argument("--train", type=float, default=0.8, help="train 비율")
    ap.add_argument("--val",   type=float, default=0.1, help="val 비율")
    ap.add_argument("--test",  type=float, default=0.1, help="test 비율")
    ap.add_argument("--seed",  type=int,   default=42,  help="난수 시드")
    ap.add_argument("--no-stratify", action="store_true",
                    help="relation 기준 계층화 끄기")
    ap.add_argument("--jsonl", action="store_true",
                    help="JSONL로 저장(기본은 JSON 배열)")
    args = ap.parse_args()

    ratios = (args.train, args.val, args.test)
    assert abs(sum(ratios) - 1.0) < 1e-6, "train/val/test 비율의 합은 1이어야 합니다."

    os.makedirs(args.outdir, exist_ok=True)

    items = read_jsonl(args.input)
    print(f"Loaded: {len(items)} samples")

    if args.no_stratify:
        tr, va, te = random_split(items, ratios, seed=args.seed)
    else:
        tr, va, te = stratified_split(items, ratios, seed=args.seed, key="relation")

    # 파일명
    f_tr = os.path.join(args.outdir, "train_set_xywh.jsonl" if args.jsonl else "train_set_xywh.json")
    f_va = os.path.join(args.outdir, "val_set_xywh.jsonl"   if args.jsonl else "val_set_xywh.json")
    f_te = os.path.join(args.outdir, "test_set_xywh.jsonl"  if args.jsonl else "test_set_xywh.json")

    # 저장
    if args.jsonl:
        write_jsonl(f_tr, tr)
        write_jsonl(f_va, va)
        write_jsonl(f_te, te)
    else:
        write_json(f_tr, tr)
        write_json(f_va, va)
        write_json(f_te, te)

    print(f"Saved: train={len(tr)} → {f_tr}")
    print(f"       val  ={len(va)} → {f_va}")
    print(f"       test ={len(te)} → {f_te}")

if __name__ == "__main__":
    main()
