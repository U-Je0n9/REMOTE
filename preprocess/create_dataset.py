import json, re, os, io, hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from statistics import mean
from tqdm import tqdm

INPUT_PATH = "data.json"   
OUTPUT_PATH = "REMOTE/datasets/remote_re.jsonl" 
IMG_DIR = "REMOTE/datasets/UMKE_IMG"     
MAX_DISTANCE = 17

TB_RE = re.compile(r"^TB-(\d+)-(\d+)$") 
IB_RE = re.compile(r"^IB-(\d+)-")       

TOK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

OBJ_INDEX: Dict[str, List[Dict[str, Any]]] = {}

def _round_xywh(xywh: List[float], ndigits: int = 5) -> List[float]:
    return [round(float(v), ndigits) for v in xywh]

def _xywh_close(a: List[float], b: List[float], eps: float = 1e-3) -> bool:
    return sum((float(x)-float(y))**2 for x, y in zip(a, b)) ** 0.5 <= eps

def get_obj_index_for_image(src_url: str, label: str, xywh: List[float]) -> int:
    """
    같은 이미지(src_url) 내에서 (label, xywh_rounded)에 안정적인 연속 인덱스를 부여.
    처음 보는 조합이면 다음 번호를 할당.
    """
    xywh_r = _round_xywh(xywh)
    bucket = OBJ_INDEX.setdefault(src_url, [])
    for rec in bucket:
        if rec['label'] == label:
            return rec['idx']
    new_idx = len(bucket)
    bucket.append({'label': label, 'xywh': xywh_r, 'idx': new_idx})
    return new_idx


def preflight():
    print("[preflight]")
    print("cwd:", os.getcwd())
    print("INPUT_PATH exists:", os.path.exists(INPUT_PATH))
    print("OUTPUT_DIR exists (before):", os.path.isdir(os.path.dirname(OUTPUT_PATH)))
    print("IMG_DIR exists (before):", os.path.isdir(IMG_DIR))

    ensure_dir(os.path.dirname(OUTPUT_PATH))
    ensure_dir(IMG_DIR)

    try:
        testfile = os.path.join(os.path.dirname(OUTPUT_PATH), "_write_test.tmp")
        with open(testfile, "w", encoding="utf-8") as f:
            f.write("ok")
        os.replace(testfile, testfile) 
        print("WRITE TEST: OK")
    except Exception as e:
        print("WRITE TEST: FAIL ->", repr(e))

    print("OUTPUT_DIR exists (after):", os.path.isdir(os.path.dirname(OUTPUT_PATH)))
    print("IMG_DIR exists (after):", os.path.isdir(IMG_DIR))
    print("OUTPUT_PATH target:", OUTPUT_PATH)

def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in TOK_RE.finditer(text or "")]

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_dataset(path: str) -> List[dict]:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        items = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    else:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

def parse_tb(tb_id: str) -> Optional[Tuple[int, int]]:
    m = TB_RE.match(tb_id or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def parse_ib(ib_id: str) -> Optional[int]:
    m = IB_RE.match(ib_id or "")
    if not m:
        return None
    return int(m.group(1))


def char_span_to_token_span(text: str, char_start: int, char_end: int) -> Optional[Tuple[int, int]]:
    spans = tokenize_with_spans(text)
    if not spans:
        return None
    start_tok = None
    end_tok = None
    for i, (tok, s, e) in enumerate(spans):
        if e <= char_start:
            continue
        if s >= char_end:
            break
        if start_tok is None:
            start_tok = i
        end_tok = i + 1  # half-open
    if start_tok is None or end_tok is None:
        return None
    return start_tok, end_tok

def _extract_images_from_turn(turn: dict):
    images = []
    msg_type = turn.get("msg_type", "")
    msg = turn.get("msg", {})

    if msg_type == "image":
        if isinstance(msg, dict):
            imgs = msg.get("images") or []
            if isinstance(imgs, list):
                images.extend(imgs)

    if isinstance(msg, dict):
        info = msg.get("info") or {}
        ik = info.get("image_knowledge") or []
        if isinstance(ik, list):
            for x in ik:
                if isinstance(x, dict):
                    images.append(x)

    return images


def _first_src_fname_from_images(images: List[dict]) -> Optional[Tuple[str, Optional[str]]]:
    for img in images:
        src = img.get("src")
        if src:
            return src, img.get("f_name")
    return None

def _find_first_image_in_dialog(dialog: List[dict]) -> Optional[Tuple[str, Optional[str]]]:
    for turn in dialog:
        images = _extract_images_from_turn(turn)
        pair = _first_src_fname_from_images(images)
        if pair:
            return pair
    return None

def _find_recent_prev_image(dialog: List[dict], real_u: int) -> Optional[Tuple[str, Optional[str]]]:
    for i in range(real_u - 1, -1, -1):
        images = _extract_images_from_turn(dialog[i])
        pair = _first_src_fname_from_images(images)
        if pair:
            return pair
    return None

def _find_in_turn_image(dialog: List[dict], real_u: int) -> Optional[Tuple[str, Optional[str]]]:
    images = _extract_images_from_turn(dialog[real_u]) if 0 <= real_u < len(dialog) else []
    return _first_src_fname_from_images(images)

def select_preferred_image(dialog: List[dict], idx_map: List[int], cu_idx: int) -> Optional[Tuple[str, Optional[str]]]:
    ru = get_real_u(idx_map, cu_idx)
    if ru is None:
        return _find_first_image_in_dialog(dialog)
    # 1
    pair = _find_in_turn_image(dialog, ru)
    if pair:
        return pair
    # 2
    pair = _find_recent_prev_image(dialog, ru)
    if pair:
        return pair
    # 3
    return _find_first_image_in_dialog(dialog)

def ensure_blank_png() -> str:
    ensure_dir(IMG_DIR)
    out_path = os.path.join(IMG_DIR, "_blank.png")
    if not os.path.exists(out_path):
        try:
            from PIL import Image
            img = Image.new("RGB", (1, 1), (255, 255, 255))
            img.save(out_path, format="PNG")
        except Exception:
            with open(out_path, "wb") as f:
                f.write(
                    bytes.fromhex(
                        "89504E470D0A1A0A0000000D4948445200000001000000010802000000907753DE0000000A4944415408D763F8FFFF3F0005FE02FEA7"
                        "9A3C000000000049454E44AE426082"
                    )
                )
    return os.path.basename(out_path)

def save_image_to_png(src_url: str, f_name: Optional[str]) -> Optional[str]:
    try:
        import requests
        from PIL import Image

        ensure_dir(IMG_DIR)
        if f_name:
            stem = str(f_name)
        else:
            h = hashlib.md5(src_url.encode("utf-8")).hexdigest()[:12]
            stem = f"url_{h}"
        out_path = os.path.join(IMG_DIR, f"{stem}.png")
        if os.path.exists(out_path):
            return os.path.basename(out_path)

        resp = requests.get(src_url, timeout=15)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img.save(out_path, format="PNG")
        return os.path.basename(out_path)
    except Exception:
        return None

def is_countable_turn(turn: dict) -> bool:
    speaker = (turn.get("speaker") or "").lower()
    msg_type = (turn.get("msg_type") or "").lower()
    return speaker in {"user", "system"} and msg_type in {"text", "image"}

def build_idx_map(dialog: List[dict]) -> List[int]:
    idx_map = []
    for i, turn in enumerate(dialog):
        if is_countable_turn(turn):
            idx_map.append(i)
    return idx_map

def get_real_u(idx_map: List[int], compact_u: int) -> Optional[int]:
    if compact_u is None or compact_u < 0 or compact_u >= len(idx_map):
        return None
    return idx_map[compact_u]

def build_segment_and_token_map(dialog: List[dict], idx_map: List[int], cu_start: int, cu_end: int):
    lines = []
    global_tokens: List[str] = []
    utter_content_token_start: Dict[int, int] = {}
    cur_tok_offset = 0

    for cu in range(cu_start, cu_end + 1):
        ru = get_real_u(idx_map, cu)
        if ru is None:
            continue
        turn = dialog[ru]
        speaker = turn.get("speaker", "")
        msg_type = turn.get("msg_type", "text")
        text = ""
        if msg_type == "text":
            m = turn.get("msg", {}) or {}
            text = m.get("text", "") or ""

        prefix = "USER:" if speaker == "user" else ("SYSTEM:" if speaker == "system" else speaker.upper() + ":")
        line = f"{prefix}{text}"
        lines.append(line)

        prefix_tokens = [t for (t, s, e) in tokenize_with_spans(prefix)]
        content_spans = tokenize_with_spans(text)
        content_tokens = [t for (t, s, e) in content_spans]

        global_tokens.extend(prefix_tokens)
        utter_content_token_start[cu] = cur_tok_offset + len(prefix_tokens)  
        global_tokens.extend(content_tokens)

        cur_tok_offset += len(prefix_tokens) + len(content_tokens)

    doc_text = "\n".join(lines)
    return doc_text, global_tokens, utter_content_token_start

def find_image_obj_and_bbox(turn: dict, label: str):
    images = _extract_images_from_turn(turn)
    for img in images:
        bbox_info = img.get("bbox_info") or {}
        if not isinstance(bbox_info, dict):
            continue
        if label in bbox_info:
            info = bbox_info.get(label) or {}
            xywh = info.get("bbox_xywh_norm") or info.get("bbox_xywh")
            if not (isinstance(xywh, list) and len(xywh) == 4):
                continue
            f_name = img.get("f_name")
            src = img.get("src")
            return label, [float(x) for x in xywh], f_name, src
    return None

def build_output_sample(dialog: List[dict],
                        idx_map: List[int],
                        head_m: dict, tail_m: dict,
                        relation_label: str) -> Optional[dict]:
    def _cu_from_m(m: dict) -> Optional[int]:
        if m.get("target") == "T":
            parsed = parse_tb(m.get("id", ""))
            return parsed[0] if parsed else None
        else:
            return parse_ib(m.get("id", ""))

    hu = _cu_from_m(head_m)
    tu = _cu_from_m(tail_m)
    if hu is None or tu is None:
        return None

    if abs(hu - tu) > MAX_DISTANCE:
        return None

    cmin, cmax = min(hu, tu), max(hu, tu)

    doc_text, tokens, content_tok_start = build_segment_and_token_map(dialog, idx_map, cmin, cmax)

    def build_side(m: dict) -> Optional[dict]:
        if m.get("target") == "T":
            tb = parse_tb(m.get("id", ""))
            if not tb:
                return None
            cu_idx, char_start = tb[0], m.get("start", 0)
            char_end = m.get("end", char_start)

            ru_idx = get_real_u(idx_map, cu_idx)
            if ru_idx is None or not (0 <= ru_idx < len(dialog)):
                return None
            turn = dialog[ru_idx]
            if (turn.get("msg_type") or "") != "text":
                return None

            turn_text = (turn.get("msg") or {}).get("text", "") or ""
            tok_span = char_span_to_token_span(turn_text, char_start, char_end)
            if not tok_span:
                return None

            local_s, local_e = tok_span
            global_start = content_tok_start.get(cu_idx, 0) + local_s
            global_end   = content_tok_start.get(cu_idx, 0) + local_e
            name = turn_text[char_start:char_end]
            return {"name": name, "pos": [int(global_start), int(global_end)], "_cu_idx": cu_idx, "_is_image": False}

        cu_idx = parse_ib(m.get("id", ""))
        if cu_idx is None:
            return None
        ru_idx = get_real_u(idx_map, cu_idx)
        if ru_idx is None or not (0 <= ru_idx < len(dialog)):
            return None

        label = m.get("label")
        turn = dialog[ru_idx]
        found = find_image_obj_and_bbox(turn, label)
        if not found:
            return None

        lbl, xywh_norm, f_name, src = found
        return {
            "name": lbl,  
            "pos": [float(xywh_norm[0]), float(xywh_norm[1]),
                    float(xywh_norm[2]), float(xywh_norm[3])],
            "_img_src": src,
            "_img_fname": f_name,
            "_cu_idx": cu_idx,
            "_is_image": True,
            "_img_label": lbl,
        }

    h_side = build_side(head_m)
    t_side = build_side(tail_m)
    if not h_side or not t_side:
        return None

    chosen_src = None
    chosen_fname = None

    for side in (h_side, t_side):
        if side.get("_is_image") and side.get("_img_src"):
            chosen_src = side["_img_src"]
            chosen_fname = side.get("_img_fname")
            break

    if not chosen_src:
        if "_cu_idx" in h_side:
            pair = select_preferred_image(dialog, idx_map, h_side["_cu_idx"])
            if pair:
                chosen_src, chosen_fname = pair
        if not chosen_src and "_cu_idx" in t_side:
            pair = select_preferred_image(dialog, idx_map, t_side["_cu_idx"])
            if pair:
                chosen_src, chosen_fname = pair

    img_id = None
    if chosen_src:
        saved = save_image_to_png(chosen_src, chosen_fname)
        if saved:
            img_id = saved
            

    if not img_id:
        img_id = ensure_blank_png()
        if not chosen_src:
            chosen_src = "__blank__"

    for side in (h_side, t_side):
        if side.get("_is_image"):
            src_for_index = side.get("_img_src") or chosen_src
            idx = get_obj_index_for_image(src_for_index, side["_img_label"], side["pos"])
            side["name"] = f"[OBJ{idx}]"

    for side in (h_side, t_side):
        for k in list(side.keys()):
            if k.startswith("_"):
                side.pop(k, None)

    return {
        "img_id": img_id,
        "text": doc_text,
        "token": tokens,
        "h": h_side,
        "t": t_side,
        "relation": relation_label if relation_label else "none",
    }


def convert_one_item(item: dict) -> List[dict]:
    out = []
    cr = item.get("cr_label")
    if not cr:
        return out
    entities = {e.get("id"): e for e in cr.get("entities", []) if isinstance(e, dict)}
    relations = cr.get("relations", []) or []

    def mentions_of(eid: str) -> List[dict]:
        ent = entities.get(eid)
        return ent.get("mentions", []) if ent else []

    dialog = item.get("dialog", []) or []
    idx_map = build_idx_map(dialog)  

    for rel in relations:
        head_id, tail_id = rel.get("head"), rel.get("tail")
        rlabel = rel.get("relation", "none")
        h_ms = mentions_of(head_id)
        t_ms = mentions_of(tail_id)
        if not h_ms or not t_ms:
            continue
        for hm in h_ms:
            for tm in t_ms:
                sample = build_output_sample(dialog, idx_map, hm, tm, rlabel)
                if sample:
                    out.append(sample)
    return out

def main():
    preflight()
    ensure_dir(os.path.dirname(OUTPUT_PATH))
    ensure_dir(IMG_DIR)

    items = load_dataset(INPUT_PATH)
    total_samples = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fw:
        for it in tqdm(items, desc="Converting dialogs", unit="dialog"):
            samples = convert_one_item(it)
            for s in samples:
                fw.write(json.dumps(s, ensure_ascii=False) + "\n")
            total_samples += len(samples)

    print(f"\n완료: 생성된 RE 샘플 수 = {total_samples}")
    print(f"출력 파일: {OUTPUT_PATH}")
    print(f"이미지 저장 폴더: {IMG_DIR}")

if __name__ == "__main__":
    main()
