import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

DATA_DIR = Path("REMOTE/datasets") 
SPLIT_FILES = ["train_set.json", "val_set.json", "test_set.json"]  
OUTPUT_PATH = Path("REMOTE/datasets/pos_umke.json")  

OBJ_PAT_PREFIX = "[OBJ"
OBJ_PAT_SUFFIX = "]"

def _is_number_list(x: Any, n: int) -> bool:
    return isinstance(x, list) and len(x) == n and all(isinstance(v, (int, float)) for v in x)

def _to_x1y1x2y2(b: List[float]) -> Optional[List[float]]:
    if not _is_number_list(b, 4):
        return None
    x, y, w, h = b
    x1n, y1n, x2n, y2n = x, y, x + w, y + h
    x1n = max(0.0, min(1.0, float(x1n)))
    y1n = max(0.0, min(1.0, float(y1n)))
    x2n = max(0.0, min(1.0, float(x2n)))
    y2n = max(0.0, min(1.0, float(y2n)))
    if x2n <= x1n or y2n <= y1n:
        return None
    return [x1n, y1n, x2n, y2n]

def _basename(img_id: Any) -> Optional[str]:
    if not img_id or not isinstance(img_id, str):
        return None
    return os.path.basename(img_id)

def _collect_from_side(side: Dict[str, Any]) -> Optional[Tuple[str, List[float]]]:
    if not isinstance(side, dict):
        return None
    name = side.get("name")
    pos = side.get("pos")
    if not (isinstance(name, str) and name.startswith(OBJ_PAT_PREFIX) and name.endswith(OBJ_PAT_SUFFIX)):
        return None
    box = _to_x1y1x2y2(pos)
    if box is None:
        return None
    return name, box

def _load_split(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    text_stripped = text.lstrip()
    if text_stripped.startswith("["):
        # JSON 배열
        return json.loads(text)
    else:
        # JSONL
        items = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
        return items

def build_pos_umke(data_dir: Path, split_files: List[str]) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {}
    for fname in split_files:
        items = _load_split(data_dir / fname)
        for it in items:
            img_id = _basename(it.get("img_id"))
            if not img_id:
                continue

            img_map = out.setdefault(img_id, {})

            for key in ("h", "t"):
                if key not in it:
                    continue
                res = _collect_from_side(it[key])
                if res is None:
                    continue
                label, box = res
                img_map[label] = box  
    return out

def main():
    pos_map = build_pos_umke(DATA_DIR, SPLIT_FILES)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pos_map, f, ensure_ascii=False, indent=2)
    print(f"Saved: {OUTPUT_PATH} ({len(pos_map)} images)")

if __name__ == "__main__":
    main()
