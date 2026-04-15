# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Race bib number detector: given a photo of runners, detect and return the race numbers (startovní čísla) visible on their bibs. A Python reimplementation of [gheinrich/bibnumber](https://github.com/gheinrich/bibnumber), replacing the original OpenCV/Tesseract pipeline with modern deep-learning OCR engines.

**Recommended Python version: 3.10 or 3.11.** PyTorch (EasyOCR) and PaddlePaddle may not support Python 3.13.

## Installing Dependencies

```bash
# Core image processing
pip install opencv-python numpy

# OCR engine – install at least one:
pip install easyocr                          # default engine
pip install paddleocr paddlepaddle           # alternative (CPU)
# pip install paddlepaddle-gpu              # alternative (GPU)

# Optional: write keywords to JPEG EXIF (Windows Explorer)
pip install piexif pillow
```

## Running the Code

```bash
# CLI – process one image
python3 bibnumber.py test_race.jpg

# CLI – process one image with PaddleOCR engine and custom output folder
python3 bibnumber.py test_race.jpg --engine paddleocr --out results/

# CLI – process a whole folder
python3 bibnumber.py slozka_s_fotkami/

# GUI application (tkinter)
python3 app.py
```

## Architecture

### Module Relationships

```
bibnumber.py  ←  app.py   (primary GUI: folder processing + IPTC metadata writing)
bibnumber.py  ←  gui.py   (legacy GUI: Tesseract-based, superseded by app.py)
```

`bibnumber.py` is the core detection library; both GUIs import `detect_bibs()` from it. `gui.py` is a legacy file left from the original Tesseract-based implementation; the active GUI is `app.py`.

### Detection Pipeline (`bibnumber.py`)

`detect_bibs(image_path, out_dir, debug, engine)` → `list[int]`

1. **Resize** – images are scaled down to 1200 px (EasyOCR) or 1024 px (PaddleOCR) on the longest dimension.
2. **OCR** – raw detections are obtained from the selected engine. Both are lazily initialised as module-level singletons (`_easyocr_reader`, `_paddleocr_reader`); first call downloads models (~100 MB for EasyOCR).
3. **Filter** – each detection must pass:
   - confidence ≥ 0.35
   - height ≥ 12 px, aspect ratio 0.4–9.0
   - text is all digits (no letters)
   - 2–6 digits long
   - not a repeating pattern (e.g. "1111")
   - value ≥ 10
4. **Visual validation** (`_looks_like_bib`) – checks that the bbox region has high contrast between dark/light pixels (≥ 35 difference in mean) and that the surrounding background is uniform (std < 65). Skipped for detections with confidence ≥ 0.7.
5. **Output** – returns `sorted(set(results))`; optionally writes an annotated JPEG to `out_dir`.

The system is tuned for **high precision over high recall**: it prefers missing a number to reporting a wrong one.

### GUI Application (`app.py`)

- Tkinter GUI; processing runs on a background `daemon` thread to keep the UI responsive.
- After detection, annotated copies are saved to `<folder>/_annotated/`.
- For JPEG files, detected numbers are written as metadata:
  - **IPTC dataset 2:25** keywords (raw byte manipulation, no external lib required) – readable by Lightroom, Capture One, exiftool.
  - **EXIF XPKeywords** (via `piexif`) – readable in Windows Explorer. Only written when `piexif` is installed.
- Engine selection (EasyOCR / PaddleOCR) is exposed in the UI; unavailable engines are disabled. EasyOCR import must happen before PaddleOCR on Windows to avoid DLL conflicts between PyTorch and PaddlePaddle.

### PaddleOCR API Compatibility

PaddleOCR changed its Python API between v2.x and v3.x. `_get_paddleocr_reader()` tries four constructor signatures in sequence (falling through on `TypeError` for unknown arguments) to support both versions transparently.

## Key Thresholds (tuning knobs)

| Parameter | Location | Effect |
|-----------|----------|--------|
| `conf < 0.35` | `detect_bibs` | Minimum OCR confidence |
| `bh_px < 12` | `detect_bibs` | Minimum bib height in pixels |
| `ratio 0.4–9.0` | `detect_bibs` | Aspect ratio guard |
| `conf < 0.7` skip visual check | `detect_bibs` | High-confidence detections bypass `_looks_like_bib` |
| contrast `< 35` | `_looks_like_bib` | Min light/dark mean difference |
| bg std `< 65` | `_looks_like_bib` | Max background uniformity (lower = stricter) |
| `max_dim` 1200/1024 | `detect_bibs` | Resize limit per engine |
