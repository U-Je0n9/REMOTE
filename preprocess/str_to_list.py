# fix_names_from_pos.py
# -*- coding: utf-8 -*-
import json, os
from copy import deepcopy

# 입력/출력 경로를 필요에 맞게 수정하세요.
DATA_DIR = "/home/ujeong/KETI/REMOTE/datasets"
INPUT_FILES = ["train_set_str.json", "val_set_str.json", "test_set_str.json"]

def slice_tokens(tokens, start_idx, end_idx):
    """토큰 리스트에서 [start_idx, end_idx) 범위를 안전하게 슬라이스 (end 미포함)."""
    n = len(tokens)
    s = max(0, int(start_idx))
    e = min(n, int(end_idx))
    if s >= e:
        return []
    return tokens[s:e]

def fix_sample(sample):
    """
    샘플의 h/t에서 pos가 길이 2(텍스트 스팬)면 name을 토큰 슬라이스(list[str])로 교체.
    pos가 길이 4(박스)면 건드리지 않음.
    """
    out = deepcopy(sample)
    tokens = out.get("token", [])
    for key in ("h", "t"):
        if key not in out or not isinstance(out[key], dict):
            continue
        pos = out[key].get("pos", None)
        if isinstance(pos, list) and len(pos) == 2 and all(isinstance(x, (int, float)) for x in pos):
            start, end = int(pos[0]), int(pos[1])
            out[key]["name"] = slice_tokens(tokens, start, end)
        # len(pos)==4 (bbox) 는 그대로 둠
    return out

def process_file(path_in, path_out):
    with open(path_in, "r", encoding="utf-8") as f:
        data = json.load(f)  # 배열 JSON 전제

    fixed = []
    changed = 0
    for s in data:
        before_h = s.get("h", {}).get("name")
        before_t = s.get("t", {}).get("name")
        fs = fix_sample(s)
        after_h = fs.get("h", {}).get("name")
        after_t = fs.get("t", {}).get("name")
        if before_h != after_h or before_t != after_t:
            changed += 1
        fixed.append(fs)

    with open(path_out, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)
    print(f"✅ {os.path.basename(path_in)} → {os.path.basename(path_out)} | 총 {len(fixed)}개, 수정 {changed}개")

def main():
    for fname in INPUT_FILES:
        path_in = os.path.join(DATA_DIR, fname)
        # `_str` 제거하고 저장
        base_name = fname.replace("_str", "")
        path_out = os.path.join(DATA_DIR, base_name)
        process_file(path_in, path_out)

if __name__ == "__main__":
    main()
