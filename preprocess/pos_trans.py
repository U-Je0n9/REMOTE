# build_pos_umke.py
# -*- coding: utf-8 -*-
"""
train.json / valid.json / test.json을 순회하며 이미지 엔티티([OBJk])의 bbox를 모아
pos_umke 형태의 사전 { img_filename: { "[OBJ0]": [x1,y1,x2,y2], ... }, ... } 로 저장.

추가 변경:
- obj가 하나도 없는 img_id도 반드시 키를 만들고 빈 dict로 남김: { "xxx.png": {} }
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# === 경로 설정 ===
DATA_DIR = Path("/home/ujeong/KETI/REMOTE/datasets")  # train/valid/test 파일이 있는 폴더
SPLIT_FILES = ["train_set.json", "val_set.json", "test_set.json"]  # 필요시 수정
OUTPUT_PATH = Path("/home/ujeong/KETI/REMOTE/datasets/pos_umke.json")  # 출력 파일 경로

OBJ_PAT_PREFIX = "[OBJ"
OBJ_PAT_SUFFIX = "]"

def _is_number_list(x: Any, n: int) -> bool:
    return isinstance(x, list) and len(x) == n and all(isinstance(v, (int, float)) for v in x)

def _to_x1y1x2y2(b: List[float]) -> Optional[List[float]]:
    """b가 [x1,y1,x2,y2]인지 [x,y,w,h]인지 판별하여 x1y1x2y2로 반환. 실패 시 None."""
    if not _is_number_list(b, 4):
        return None
    # xywh로 간주하여 변환 (이미 x1y1x2y2면도 통과 가능)
    x, y, w, h = b
    x1n, y1n, x2n, y2n = x, y, x + w, y + h
    # 0~1 클램프
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
    """
    h / t 딕셔너리에서 [OBJk]와 bbox를 추출하여 (label, [x1,y1,x2,y2]) 반환.
    해당되지 않으면 None.
    """
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
    """
    .json(리스트) 또는 .jsonl(줄당 객체) 모두 지원
    """
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
                # 이미지가 없으면 스킵(텍스트 전용 샘플)
                continue

            # ✅ obj가 없어도 비어있는 엔트리를 먼저 만들어 둔다
            img_map = out.setdefault(img_id, {})

            # h / t 에 대해 시도 (있다면 갱신)
            for key in ("h", "t"):
                if key not in it:
                    continue
                res = _collect_from_side(it[key])
                if res is None:
                    continue
                label, box = res
                img_map[label] = box  # 동일 키가 여러 번 나오면 마지막 값 채택
    return out

def main():
    pos_map = build_pos_umke(DATA_DIR, SPLIT_FILES)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pos_map, f, ensure_ascii=False, indent=2)
    print(f"Saved: {OUTPUT_PATH} ({len(pos_map)} images)")

if __name__ == "__main__":
    main()
