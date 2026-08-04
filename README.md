# CartGate

셀프 계산대 매장을 위한 AI 출구 게이트. 손님이 휴대폰으로 스캔·결제하고, 출구에서 카메라가
카트를 보고 **보이는 물건이 결제 내역과 일치하는지** 확인합니다. 카트 안을 전부 인식하는 게
아니라 **영수증 조건부 이상탐지(receipt-conditioned anomaly detection)** — 미결제·수량 초과만
잡으면 되므로, 보이는 물건을 전체 카탈로그가 아니라 *그 카트의 영수증 SKU*와만 대조합니다.

## 파이프라인

```
 카메라 3대 (상단 1 + 측면 2, ~30fps)
        │
        ▼
 [1] 검출(Detector)     YOLO11n, "product" 단일 클래스   → 박스 (클래스 무관)
        │
        ▼
 [2] 인식(Recognition)  DINOv2-S/14 + ArcFace 헤드 → 256-d 임베딩,
        │               영수증 SKU에만 대조
        ▼
 [3] 집계(Aggregation)  다중 프레임 추적 + 3카메라 융합 → SKU별 수량
        │
        ▼
 [4] 판정(Decision)     결제 DB와 대조 → PASS / FLAG / REVIEW
```

**[1]~[3]은 이 리포에 구현·학습되어 있습니다. [4] 판정 레이어는 팀원 담당** —
[`docs/DECISION_LAYER.md`](docs/DECISION_LAYER.md) 참고.

## 디렉토리 구조

```
cjs/
├─ cartgate/              # 라이브러리 (import 되는 모듈)
│   ├─ embed.py           #   임베딩 백엔드 (classical + ONNX)
│   ├─ gallery.py         #   SKU별 임베딩 갤러리 생성
│   ├─ match.py           #   크롭↔갤러리 매칭 · 영수증 대조 · 판정
│   ├─ synth.py           #   검출기 학습용 합성 카트 장면
│   ├─ segment.py         #   누끼 매팅 (rembg / grabcut)
│   └─ train_embed.py     #   MobileNetV3 베이스라인 + 공용 augmentation 유틸
│
├─ scripts/               # 실행 진입점 (리포 루트에서 실행)
│   ├─ ingest.py                # 원본 사진 → dataset/
│   ├─ make_detect_data.py      # 검출기 학습 데이터 생성
│   ├─ train_detector.py        # 검출기 학습 (YOLO)
│   ├─ train_recognition.py     # 인식 임베더 학습 → dino_arc.onnx
│   ├─ make_paired.py           # 원본 + BiRefNet 누끼 pair 생성
│   ├─ pipeline.py              # 엔드투엔드 데모 (검출→인식→융합→판정)
│   ├─ viz_recognition.py       # 인식 결과 시각화
│   ├─ stress_test.py           # 강건성 스윕 (혼잡도 × 화질)
│   ├─ build_handoff.py         # 백엔드용 상품 마스터 (xlsx/csv)
│   └─ export_deploy.py         # Jetson용 ONNX export
│
├─ docs/DECISION_LAYER.md  # 판정 레이어 스펙 (팀원용)
├─ products.csv            # 상품 마스터 (sku_id · 상품명 · EAN-13 바코드)
├─ pyproject.toml          # 패키지 정의 (pip install -e .)
├─ requirements.txt
└─ (데이터·모델 디렉토리 — git 제외, 별도 전달)
```

> 데이터·모델·산출물은 `.gitignore` 처리되어 git에 안 올라갑니다 (**데이터** 항목 참고).

## 설치

이 환경 드라이버가 CUDA 12.6이라, 맞는 torch로 conda 환경을 씁니다:

```bash
conda create -n cartgate python=3.11 -y
conda activate cartgate
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .          # cartgate 패키지를 import 가능하게 등록
```

CPU만으로도 실행 가능(‑‑index-url 빼고 `onnxruntime` 사용). 학습은 GPU 필요.
스크립트는 **리포 루트에서 실행**하세요 (`dataset/`, `dino_arc.onnx` 등 경로가 상대 경로).

## 데이터

용량이 커서 git에 없습니다. 전달받은 번들을 리포 루트에 풀면 아래 구조가 됩니다:

```
dataset/
  images/<sku_id>__<이름>/*.jpg     상품 사진 (51 SKU) — 갤러리 소스
  barcode/<sku_id>__<이름>.png      상품 바코드
out/cut_rembg/                      인식 학습용 누끼 (RGBA)
detect_data/                        검출기 학습용 합성 데이터
runs/detect/.../weights/best.pt     학습된 검출기
dino_arc.onnx                       학습된 인식 임베더
yolo11n.pt                          검출기 베이스 가중치
```

`products.csv`(sku_id·상품명·EAN-13 바코드)는 상품 마스터로 git에 포함됩니다.

## 실행

```bash
# 엔드투엔드 데모 (검출 → 인식 → 융합 → 판정)
python scripts/pipeline.py

# 어려운 장면(가림/저해상도/블러/저조도) 인식 시각화  → out/viz/
python scripts/viz_recognition.py

# 강건성 스윕: 혼잡도 × 화질별 인식/e2e
python scripts/stress_test.py

# 백엔드용 상품 마스터 (xlsx + csv)  → service_products/
python scripts/build_handoff.py --no-images
```

재학습 (GPU):

```bash
python scripts/make_detect_data.py && python scripts/train_detector.py   # 검출기
python scripts/train_recognition.py                                      # 인식 → dino_arc.onnx
```

## 현재 성능 (합성 데이터 기준)

- **검출기**: mAP50 ≈ 0.94, mAP50-95 ≈ 0.80
- **인식** (실제 검출 크롭, 영수증 제한 top-1): **90.8%**
  (DINOv2+ArcFace + 합성뷰 보강 갤러리; MobileNetV3 베이스라인은 81.8%)
- **스트레스** (인식 top-1): 깨끗 82-94%, 저해상도 75-91%, 최악 62-77% (혼잡도별)

전부 합성 카트 기준입니다. 가장 큰 남은 과제는 **실제 게이트 영상으로의 검증과 임계값 보정**.

## 참고

- 인식은 *영수증 제한*: 크롭을 그 카트 영수증 SKU와만 대조 → SKU당 1~3장만으로도 동작.
- 런타임엔 배경 제거 안 함. 누끼(BiRefNet)는 오프라인 학습 데이터 합성용으로만 사용
  (물체를 다양한 배경에 얹어 배경 강건성 확보).
- 배포 타깃은 Jetson(AGX/Orin) + TensorRT. `export_deploy.py`가 검출기 ONNX를 뽑고,
  `dino_arc.onnx`가 인식 임베더입니다.
