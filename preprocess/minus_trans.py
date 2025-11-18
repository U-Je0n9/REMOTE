# fix_negative_bbox.py
# -*- coding: utf-8 -*-
import json, os
from copy import deepcopy

DATA_DIR = "/home/ujeong/KETI/REMOTE/datasets"
INPUT_FILES = ["train_set.json", "val_set.json", "test_set.json"]
OUTPUT_SUFFIX = "_bbox_fixed.json"  # 예: train_set_bbox_fixed.json

def clamp_bbox(pos):
    """pos(리스트)에 음수값이 있으면 0으로 치환."""
    if not isinstance(pos, list):
        return pos
    return [x if isinstance(x, (int, float)) and x >= 0 else 0 for x in pos]

def fix_sample(sample):
    """샘플의 h/t.pos에서 음수 좌표를 0으로 교체."""
    out = deepcopy(sample)
    for key in ("h", "t"):
        if key not in out or not isinstance(out[key], dict):
            continue
        pos = out[key].get("pos")
        if isinstance(pos, list) and len(pos) == 4:
            fixed = clamp_bbox(pos)
            if fixed != pos:
                out[key]["pos"] = fixed
    return out

def process_file(path_in, path_out):
    with open(path_in, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixed = []
    changed = 0
    for s in data:
        before_h = s.get("h", {}).get("pos")
        before_t = s.get("t", {}).get("pos")
        fs = fix_sample(s)
        after_h = fs.get("h", {}).get("pos")
        after_t = fs.get("t", {}).get("pos")
        if before_h != after_h or before_t != after_t:
            changed += 1
        fixed.append(fs)

    with open(path_out, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)

    print(f"✅ {os.path.basename(path_in)} → {os.path.basename(path_out)} | 총 {len(fixed)}개, 수정 {changed}개")

def main():
    for fname in INPUT_FILES:
        path_in = os.path.join(DATA_DIR, fname)
        path_out = os.path.join(DATA_DIR, fname.replace(".json", OUTPUT_SUFFIX))
        process_file(path_in, path_out)

if __name__ == "__main__":
    main()
