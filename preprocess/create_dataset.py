"""
대화형 데이터셋(cr_label 포함)을 REMOTE 스타일 RE 데이터로 변환하는 스크립트.
- 거리 필터: |u_head - u_tail| <= MAX_DISTANCE (기본 17)
- 모든 멘션 조합 출력
- 텍스트는 USER/SYSTEM 라벨을 줄바꿈 포함으로 이어붙인 하나의 문서로 만들고 token/pos 생성
- 이미지 멘션은 [OBJk] 네이밍 + bbox_xywh_norm 그대로 pos 사용
- 이미지 파일은 PNG로 /home/ujeong/tmp/REMOTE/our_datasets/IMG 에 저장하고 img_id에 파일명 기록
"""

import json, re, os, io, hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from statistics import mean
from tqdm import tqdm

# --- 설정 ---
INPUT_PATH = "/home/ujeong/KETI/data.json"     # 입력 JSON 또는 JSONL
OUTPUT_PATH = "/home/ujeong/KETI/REMOTE/datasets/remote_re.jsonl"  # 출력 JSONL
IMG_DIR = "/home/ujeong/KETI/REMOTE/datasets/UMKE_IMG"       # 이미지 저장 폴더
MAX_DISTANCE = 17

# --- 정규식: mention id 파싱 ---
TB_RE = re.compile(r"^TB-(\d+)-(\d+)$")  # TB-u-idx   (u=발화 인덱스 0부터, idx=문자 위치)
IB_RE = re.compile(r"^IB-(\d+)-")        # IB-u-...   (u=발화 인덱스 0부터)

# --- 토크나이저: 공백/문장부호 기준 (유니코드) ---
# 토큰과 (start_char, end_char) 스팬을 함께 반환
TOK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# --- 안정적인 OBJ 인덱싱 전역 매핑 ---
# src_url -> list of dicts [{'label': str, 'xywh': [x,y,w,h], 'idx': int}]
OBJ_INDEX: Dict[str, List[Dict[str, Any]]] = {}

def _round_xywh(xywh: List[float], ndigits: int = 5) -> List[float]:
    return [round(float(v), ndigits) for v in xywh]

def _xywh_close(a: List[float], b: List[float], eps: float = 1e-3) -> bool:
    # 좌표가 아주 근소하게 다른 경우를 같은 객체로 취급 (필요시 eps 조절)
    return sum((float(x)-float(y))**2 for x, y in zip(a, b)) ** 0.5 <= eps

def get_obj_index_for_image(src_url: str, label: str, xywh: List[float]) -> int:
    """
    같은 이미지(src_url) 내에서 (label, xywh_rounded)에 안정적인 연속 인덱스를 부여.
    처음 보는 조합이면 다음 번호를 할당.
    """
    xywh_r = _round_xywh(xywh)
    bucket = OBJ_INDEX.setdefault(src_url, [])
    # 1) 완전 일치(라벨+좌표 근사치) 재사용
    for rec in bucket:
        if rec['label'] == label:
            return rec['idx']
    # 2) 라벨만 같고 좌표가 미세하게 다른 경우 → 좌표가 충분히 다르면 새 객체로 취급
    new_idx = len(bucket)  # 0부터 연속 부여
    bucket.append({'label': label, 'xywh': xywh_r, 'idx': new_idx})
    return new_idx


def preflight():
    print("[preflight]")
    print("cwd:", os.getcwd())
    print("INPUT_PATH exists:", os.path.exists(INPUT_PATH))
    print("OUTPUT_DIR exists (before):", os.path.isdir(os.path.dirname(OUTPUT_PATH)))
    print("IMG_DIR exists (before):", os.path.isdir(IMG_DIR))

    # 디렉토리 보장
    ensure_dir(os.path.dirname(OUTPUT_PATH))
    ensure_dir(IMG_DIR)

    # 출력 폴더에 쓰기 가능한지 ‘시험 파일’로 확인
    try:
        testfile = os.path.join(os.path.dirname(OUTPUT_PATH), "_write_test.tmp")
        with open(testfile, "w", encoding="utf-8") as f:
            f.write("ok")
        os.replace(testfile, testfile)  # 권한 확인용 no-op
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
    """
    문자열 text에서 [char_start, char_end) 범위를 커버하는 토큰 인덱스 범위(start_tok, end_tok)를 반환.
    토큰 인덱스는 해당 텍스트 내부 기준(0부터). 없으면 None.
    """
    spans = tokenize_with_spans(text)
    if not spans:
        return None
    # 영역과 교차하는 첫 토큰/마지막 토큰 탐색
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
    """turn에서 image 리스트를 최대한 유연하게 뽑아낸다."""
    images = []
    msg_type = turn.get("msg_type", "")
    msg = turn.get("msg", {})

    # case A: 정석 image 턴
    if msg_type == "image":
        if isinstance(msg, dict):
            imgs = msg.get("images") or []
            if isinstance(imgs, list):
                images.extend(imgs)

    # case B: 텍스트 턴의 info.image_knowledge에도 이미지 정보가 들어있는 경우
    if isinstance(msg, dict):
        info = msg.get("info") or {}
        ik = info.get("image_knowledge") or []
        if isinstance(ik, list):
            # image_knowledge의 각 항목을 images와 유사한 dict로 해석
            for x in ik:
                if isinstance(x, dict):
                    images.append(x)

    return images

# -------- 이미지 선택 유틸 (항상 img_id 보장) --------

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
    # real_u 이전에서 가장 최근 이미지
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
    """
    우선순위:
      1) 해당 발화(cu_idx)의 턴에 포함된 이미지
      2) 그 이전 턴들 중 가장 최근 이미지
      3) 대화 내 가장 처음 이미지
    """
    ru = get_real_u(idx_map, cu_idx)
    if ru is None:
        # compact 매핑 실패 시 대화 첫 이미지라도 반환
        return _find_first_image_in_dialog(dialog)
    # 1순위
    pair = _find_in_turn_image(dialog, ru)
    if pair:
        return pair
    # 2순위
    pair = _find_recent_prev_image(dialog, ru)
    if pair:
        return pair
    # 3순위
    return _find_first_image_in_dialog(dialog)

def ensure_blank_png() -> str:
    """
    대화에 이미지가 전혀 없거나 다운로드 실패 시 사용할 1x1 빈 PNG 생성하여 파일명 반환.
    """
    ensure_dir(IMG_DIR)
    out_path = os.path.join(IMG_DIR, "_blank.png")
    if not os.path.exists(out_path):
        try:
            from PIL import Image
            img = Image.new("RGB", (1, 1), (255, 255, 255))
            img.save(out_path, format="PNG")
        except Exception:
            # PIL이 없으면 바이트 시그니처로 간단 PNG 생성 시도
            with open(out_path, "wb") as f:
                f.write(
                    bytes.fromhex(
                        "89504E470D0A1A0A0000000D4948445200000001000000010802000000907753DE0000000A4944415408D763F8FFFF3F0005FE02FEA7"
                        "9A3C000000000049454E44AE426082"
                    )
                )
    return os.path.basename(out_path)

def save_image_to_png(src_url: str, f_name: Optional[str]) -> Optional[str]:
    """
    src_url 이미지를 PNG로 IMG_DIR에 저장하고 파일명(예: 1205811.png)을 반환.
    - f_name이 있으면 우선 사용, 없으면 URL 해시 사용
    - 네 환경에서 인터넷이 막혀 있으면 이 함수는 실패할 수 있음 → 그 경우 None 반환
    """
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
        # 이미 있으면 재다운로드 생략
        if os.path.exists(out_path):
            return os.path.basename(out_path)

        resp = requests.get(src_url, timeout=15)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img.save(out_path, format="PNG")
        return os.path.basename(out_path)
    except Exception:
        return None

# --- 발화 카운트 기준: user/system 의 text/image 만 포함 ---
def is_countable_turn(turn: dict) -> bool:
    speaker = (turn.get("speaker") or "").lower()
    msg_type = (turn.get("msg_type") or "").lower()
    return speaker in {"user", "system"} and msg_type in {"text", "image"}

def build_idx_map(dialog: List[dict]) -> List[int]:
    """
    compact_u -> real_u 매핑 리스트.
    session_seperator 등은 건너뛰고 user/system의 text/image만 카운트.
    """
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
    """
    compact 인덱스 구간 [cu_start, cu_end]만으로 문서/토큰을 구성.
    반환되는 utter_content_token_start는 compact_u를 key로 사용.
    """
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
        utter_content_token_start[cu] = cur_tok_offset + len(prefix_tokens)  # compact_u 기준
        global_tokens.extend(content_tokens)

        cur_tok_offset += len(prefix_tokens) + len(content_tokens)

    doc_text = "\n".join(lines)
    return doc_text, global_tokens, utter_content_token_start

def find_image_obj_and_bbox(turn: dict, label: str):
    """
    주어진 turn(텍스트 턴의 image_knowledge 포함)에서 label의 bbox_xywh_norm을 찾아 반환.
    인덱스는 여기서 정하지 않는다.
    반환: (label, [x,y,w,h], f_name, src) 또는 None
    """
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
    # --- compact u 인덱스 파싱 ---
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

    # --- 거리 필터 ---
    if abs(hu - tu) > MAX_DISTANCE:
        return None

    cmin, cmax = min(hu, tu), max(hu, tu)

    # --- 문서(text/token) 구성 ---
    doc_text, tokens, content_tok_start = build_segment_and_token_map(dialog, idx_map, cmin, cmax)

    def build_side(m: dict) -> Optional[dict]:
        if m.get("target") == "T":
            # 텍스트 멘션
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

        # 이미지 멘션
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
            "name": lbl,  # 임시: 나중에 [OBJk]로 치환
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

    # --- img_id 선택/저장 (단일 블록, 빈 PNG 보장) ---
    chosen_src = None
    chosen_fname = None

    # A. 사이드가 직접 참조하는 이미지가 있으면 우선 사용
    for side in (h_side, t_side):
        if side.get("_is_image") and side.get("_img_src"):
            chosen_src = side["_img_src"]
            chosen_fname = side.get("_img_fname")
            break

    # B. 없으면 우선순위 규칙(해당 발화 → 이전 최근 → 대화 첫 이미지)
    if not chosen_src:
        if "_cu_idx" in h_side:
            pair = select_preferred_image(dialog, idx_map, h_side["_cu_idx"])
            if pair:
                chosen_src, chosen_fname = pair
        if not chosen_src and "_cu_idx" in t_side:
            pair = select_preferred_image(dialog, idx_map, t_side["_cu_idx"])
            if pair:
                chosen_src, chosen_fname = pair

    # C. 저장 시도
    img_id = None
    if chosen_src:
        saved = save_image_to_png(chosen_src, chosen_fname)
        if saved:
            img_id = saved

    # D. 그래도 실패하면 빈 PNG 보장
    if not img_id:
        img_id = ensure_blank_png()
        # 빈 PNG를 위한 가상 src 키
        if not chosen_src:
            chosen_src = "__blank__"

    # --- 이미지 OBJ 인덱스 안정화 치환 ---
    # 두 사이드 모두 이미지 멘션이면, "선택된 이미지" 기준으로 OBJ 인덱스를 부여
    for side in (h_side, t_side):
        if side.get("_is_image"):
            src_for_index = side.get("_img_src") or chosen_src
            idx = get_obj_index_for_image(src_for_index, side["_img_label"], side["pos"])
            side["name"] = f"[OBJ{idx}]"

    # 내부 키 정리
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
    """
    단일 대화 샘플에서 cr_label.relations를 순회하며
    모든 (head_mention, tail_mention) 조합을 생성.
    거리 필터는 compact 인덱스 기준.
    """
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
    idx_map = build_idx_map(dialog)  # compact -> real 매핑

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

    # tqdm으로 전체 아이템 처리 진행률 표시
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
