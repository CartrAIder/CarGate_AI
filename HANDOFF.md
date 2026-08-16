# Handoff

Code is on GitHub; data and models are too large for git and ship as separate
cloud zips. Below is what to send and to whom.

## Deliverables

| # | Item | Size | Contents | For |
|---|------|------|----------|-----|
| 1 | GitHub repo | — | all code (`cartgate/`, `scripts/`, `docs/`, configs, `products.csv`) | everyone |
| 2 | `cart_dataset.zip` | ~360M | 2-camera cart images (6000 = 3000/direction): `carts.json` (decision benchmark) + YOLO `labels/` + `data.yaml` (detector training) | decision-layer teammate · detector training |
| 3 | `cjs_data_bundle.zip` | ~460M | `dataset/` (gallery) + models (`dino_arc.onnx`, detector `best.pt`, `yolo11n.pt`) + `out/cut_rembg/` (recognition-training cutouts) + `products.csv` | running / retraining the vision pipeline |
| 4 | `service_products.zip` | small | product master (barcode · name · category) | backend teammate |

## Setup for a recipient

```bash
git clone <repo> && cd <repo>
conda create -n cartgate python=3.11 -y && conda activate cartgate
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt && pip install -e .
# unzip the bundles you were given into the repo root, then:
python scripts/pipeline.py                 # needs bundle #3
```

## Who needs what

- **Decision-layer teammate**: #1 + #2 (develop reconciliation on `carts.json` GT).
  Add #3 if they want to run the real detector/recognizer to get live model output.
  Start at `docs/DECISION_LAYER.md`.
- **Detector / vision (retraining)**: #1 + #2 (`yolo detect train ... data=cart_dataset/data.yaml`)
  + #3 for the gallery, recognition cutouts and current models.
- **Backend**: #4 (product master) — `sku_id` links it to the vision output.

Regenerate the cart dataset: `python scripts/make_cart_dataset.py --num 3000 --cameras 2`.
