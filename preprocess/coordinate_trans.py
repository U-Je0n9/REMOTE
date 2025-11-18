import json, os
from pathlib import Path

base = Path("/home/ujeong/KETI/REMOTE/datasets")
splits = ["train", "val", "test"]

def xywh_to_x1y1x2y2(pos):
    if isinstance(pos, list) and len(pos) == 4 and all(isinstance(v, (int, float)) for v in pos):
        x, y, w, h = pos
        return [x, y, x + w, y + h]
    return None

def convert_item(item):
    for key in ("t", "h"):
        obj = item.get(key, {})
        new_pos = xywh_to_x1y1x2y2(obj.get("pos"))
        if new_pos is not None:
            obj["pos"] = new_pos
            item[key] = obj
    return item

for split in splits:
    in_path = base / f"{split}_set_xywh.json"
    out_path = base / f"{split}_set_str.json"
    if not in_path.exists():
        print(f"SKIP: {in_path}")
        continue
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    data = [convert_item(it) for it in data]
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"변환 완료: {out_path}")
