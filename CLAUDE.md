# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project fine-tunes OpenAI's CLIP (ViT-B/32) model to analyze VRChat world thumbnail images. The trained model enables searching for similar VRChat worlds by image similarity. Training pairs world thumbnails with "(author) (title)" text strings.

## Development Environment

All development runs inside a Docker container with GPU support:

```bash
docker compose run --rm develop /bin/bash
```

Inside the container, the project is mounted at `/app`. All scripts expect paths like `/app/data/`, `/app/src/`, etc.

## Common Commands

**Download world thumbnail images:**
```bash
python /app/tools/download.py --worlds /app/data/best_worlds.csv --limit 5000
```

**Train the CLIP model:**
```bash
python /app/src/train.py --train --limit 100000
```

**Evaluate / search by image (fast, uses cached embeddings):**
```bash
python /app/src/evaluate.py --image /app/data/evaluate/test.jpg --text_embeds /app/data/worlds.text.pt
```

**Batch evaluation:**
```bash
bash /app/src/evaluate.sh
```

## Architecture

### Data Flow

1. `sql/select_best_worlds.sql` — BigQuery query that produces `data/best_worlds.csv` (top 30,000 VRChat worlds ranked by `favorites + SQRT(visits)`)
2. `tools/download.py` — Downloads thumbnails to `data/images/{wid[-2:]}/{wid}/{filename}.png`
3. `src/train.py` — Fine-tunes CLIP on image+text pairs; saves model to `tuned/vrchat-worlds/` and processor to `tuned/processor/`; also saves cached text embeddings to `data/worlds.text.pt`
4. `src/evaluate.py` — Two modes:
   - Full inference: loads both CLIP vision and text encoders
   - Fast inference: loads only `CLIPVisionModelWithProjection` + pre-computed `worlds.text.pt`

### Key Source Files

- [src/util.py](src/util.py) — `load_worlds()` (CSV parsing), `load_good_image()` (center-crop to 224×224), `load_resized_image()`
- [src/train.py](src/train.py) — `TitleAndImage` dataset class; AdamW optimizer (lr=1e-5), ExponentialLR scheduler (gamma=0.99), batch size 128, 20 epochs; train/val split is modulo-based (1/100 for validation)
- [src/evaluate.py](src/evaluate.py) — `search_title()` and `search_title_using_embeds()`; outputs top-5 matches by cosine similarity/distance
- [tools/download.py](tools/download.py) — Uses mobile User-Agent; rate-limited at 0.3s per request; validates HTTP 200 + image content-type

### Model Details

- Base model: `openai/clip-vit-base-patch32`
- Text input format: `"{author} {title}"` for each world
- Device: auto-detected CUDA or CPU (`DEVICE` variable)
- Saved models use Hugging Face `save_pretrained()` format

## Data Paths (inside container)

| Path | Contents |
|------|----------|
| `/app/data/best_worlds.csv` | World metadata: id, author_name, name, thumbnail_image_url, description |
| `/app/data/images/` | Downloaded PNG thumbnails |
| `/app/data/worlds.text.pt` | Cached CLIP text embeddings (PyTorch tensor) |
| `/app/tuned/vrchat-worlds/` | Fine-tuned CLIP model weights |
| `/app/tuned/processor/` | Saved CLIP image processor |
