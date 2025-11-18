import json, os
from copy import deepcopy

DATA_DIR = "REMOTE/datasets"
INPUT_FILES = ["train_set_str.json", "val_set_str.json", "test_set_str.json"]

def slice_tokens(tokens, start_idx, end_idx):
    n = len(tokens)
    s = max(0, int(start_idx))
    e = min(n, int(end_idx))
    if s >= e:
        return []
    return tokens[s:e]

def fix_sample(sample):
    out = deepcopy(sample)
    tokens = out.get("token", [])
    for key in ("h", "t"):
        if key not in out or not isinstance(out[key], dict):
            continue
        pos = out[key].get("pos", None)
        if isinstance(pos, list) and len(pos) == 2 and all(isinstance(x, (int, float)) for x in pos):
            start, end = int(pos[0]), int(pos[1])
            out[key]["name"] = slice_tokens(tokens, start, end)
    return out

def process_file(path_in, path_out):
    with open(path_in, "r", encoding="utf-8") as f:
        data = json.load(f) 

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
    print(f"{os.path.basename(path_in)} → {os.path.basename(path_out)} | 총 {len(fixed)}개, 수정 {changed}개")

def main():
    for fname in INPUT_FILES:
        path_in = os.path.join(DATA_DIR, fname)
        base_name = fname.replace("_str", "")
        path_out = os.path.join(DATA_DIR, base_name)
        process_file(path_in, path_out)

if __name__ == "__main__":
    main()
