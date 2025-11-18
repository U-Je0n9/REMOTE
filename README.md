# REMOTE 학습 가이드
---

이 문서는 커스텀 데이터셋을 UMRE 형식으로 준비하여 REMOTE 모델을 학습하는 전과정을 다룹니다.

---

## 주요 디렉터리

| 경로 | 설명 |
|------|------|
| `REMOTE/preprocess` | 데이터셋 변환 스크립트. `run_all.py`로 전체 파이프라인 실행 |
| `ROMOTE_code` | 학습/평가 스크립트 및 유틸리티 |
| `Depth-Anything-V2` | DepthMap 생성을 위한 의존성 |

---

## 환경 설정

```bash
pip install -r requirements.txt
```

### DepthMap 생성
```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2
# 환경 및 체크포인트는 공식 가이드를 따르세요.

# REMOTE 루트로 복귀
python ROMOTE_code/depth_data/test.py
```

---

## 데이터 전처리 파이프라인

아래 명령 한 번으로 모든 단계를 순차 실행합니다.
```bash
cd REMOTE/preprocess
python run_all.py
# 예시 옵션:
#   python run_all.py --skip split --skip minus_trans
```

| 순서 | 스크립트 | 역할 |
|------|----------|------|
| 1 | `create_dataset.py` | 대화 JSON(L) + `cr_label` → `remote_re.jsonl` 생성 |
| 2 | `split.py` | 계층적 분할 → `train/val/test_set_xywh.json` |
| 3 | `coordinate_trans.py` | `[x, y, w, h]` → `[x1, y1, x2, y2]` 변환 (`*_set_str.json`) |
| 4 | `str_to_list.py` | 텍스트 엔티티 name을 토큰 슬라이스로 교체 (`*_set.json`) |
| 5 | `minus_trans.py` | 박스 좌표 음수값 0으로 클램프 (`*_set_bbox_fixed.json`) |
| 6 | `pos_trans.py` | 이미지별 `[OBJk]` → bbox 매핑(`pos_umke.json`) 구축 |

**커스텀 데이터셋 통계** (관계별 50% 길이 내에서 최대 3개 멘션 사용용):

| 분할 | 샘플 수 |
|------|-------:|
| Train | 92,064 |
| Val   | 11,520 |
| Test  | 11,520 |
| **Total** | **115,079** |

---

## 학습 & 테스트

### 하이퍼파라미터
- `epochs = 3`
- `batch_size = 16`
- `learning_rate = 1e-5`

### 학습 실행
```bash
bash ROMOTE_code/run_umke_best.sh
```

### 테스트 / 추론
```bash
python ROMOTE_code/run_umke_best.py \
  --do_test \
  --test_path /path/to/test_set_bbox_fixed.json \
  --batch_size 16 \
  --init_checkpoint ckpt/UMKE_16_1e-05__20251029_105012
```
실행하면 클래스별 정밀도/재현율/F1을 출력하고, 결과 요약은 `test.txt`, 체크포인트는 `ckpt/`에 저장됩니다.

---

## 실험 결과 (커스텀 데이터셋)

```
precision  recall  f1-score  support
attr     0.9991   0.9993    0.9992    8059
part     1.0000   0.9524    0.9756      21
up_down  0.9982   0.9982    0.9982    3428

accuracy = 0.9989 on 11,508 samples
macro avg = (P 0.9991, R 0.9833, F1 0.9910)
weighted avg = (P 0.9989, R 0.9989, F1 0.9989)
```
