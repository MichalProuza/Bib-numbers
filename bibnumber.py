#!/usr/bin/env python3
"""
bibnumber.py – Python reimplementace projektu gheinrich/bibnumber
Rozpoznává startovní čísla ze závodních fotek.

Pipeline:
  1. Předzpracování (grayscale, denoising)
  2. Canny edge detection
  3. Stroke Width Transform (SWT) – lokalizuje oblasti textu
  4. Shlukování SWT komponent → kandidáti na text
  5. Filtrace kandidátů (aspect ratio, hustota, velikost)
  6. Groupování sousedních textových oblastí → kandidáti na číslice
  7. Rotace + ořez oblasti
  8. OCR přes Tesseract (pouze číslice)
  9. Uložení výřezů + výpis čísel

Použití:
  python3 bibnumber.py foto.jpg
  python3 bibnumber.py slozka_s_fotkami/
  python3 bibnumber.py foto.jpg --debug      # zobrazí mezikroky
  python3 bibnumber.py foto.jpg --out vystup/
"""

import cv2
import numpy as np
import pytesseract
from pytesseract import TesseractNotFoundError
import os
import sys
import argparse
import math
import shutil
from pathlib import Path
from scipy import ndimage

# Windows: automatická detekce Tesseract OCR v běžném umístění
if sys.platform == "win32":
    _win_tesseract = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if _win_tesseract.is_file():
        pytesseract.pytesseract.tesseract_cmd = str(_win_tesseract)


# ---------------------------------------------------------------------------
# 1.  STROKE WIDTH TRANSFORM
# ---------------------------------------------------------------------------

def stroke_width_transform(edges: np.ndarray, gradient_x: np.ndarray, gradient_y: np.ndarray) -> np.ndarray:
    """
    Zjednodušená SWT: pro každý hranový pixel sleduje paprsek ve směru
    normály a hledá protilehlou hranu. Výsledkem je mapa šířky tahů.
    """
    h, w = edges.shape
    swt = np.full((h, w), np.inf, dtype=np.float32)

    # Normalizované gradienty
    mag = np.sqrt(gradient_x**2 + gradient_y**2) + 1e-6
    gx = gradient_x / mag
    gy = gradient_y / mag

    rays = []

    # Pro každý hranový pixel spusť paprsek
    edge_pts = np.argwhere(edges > 0)
    for (r, c) in edge_pts:
        dx, dy = gx[r, c], gy[r, c]
        ray = [(r, c)]
        # Sleduj paprsek max 50 kroků
        for step in range(1, 50):
            nr = int(round(r + dy * step))
            nc = int(round(c + dx * step))
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                break
            if edges[nr, nc] > 0:
                # Protilehlá hrana – ověř, že gradient míří opačně
                ddx, ddy = gx[nr, nc], gy[nr, nc]
                if abs(dx * ddx + dy * ddy) < 0.8:  # cos(θ) < 0.8 → skoro opačný
                    stroke_w = math.hypot(nr - r, nc - c)
                    for (pr, pc) in ray:
                        swt[pr, pc] = min(swt[pr, pc], stroke_w)
                    swt[nr, nc] = min(swt[nr, nc], stroke_w)
                    rays.append((ray, stroke_w))
                break
            ray.append((nr, nc))

    # Nahraď inf nulou
    swt[swt == np.inf] = 0
    return swt


def swt_fast(gray: np.ndarray) -> np.ndarray:
    """
    Rychlá aproximace SWT pomocí distance transform na hranách.
    Méně přesná než plná SWT, ale řádově rychlejší pro velké fotky.
    """
    edges = cv2.Canny(gray, 50, 150)
    # Distance transform dává vzdálenost každého pixelu k nejbližší hraně
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    # SWT ≈ 2× vzdálenost ke střední ose (skeleton)
    swt = dist * 2.0
    # Maskuj jen oblasti u hran (text je hustý na hrany)
    swt[edges == 0] = 0
    return swt.astype(np.float32)


# ---------------------------------------------------------------------------
# 2.  DETEKCE TEXTOVÝCH KOMPONENT
# ---------------------------------------------------------------------------

def find_text_candidates(gray: np.ndarray, debug: bool = False):
    """
    Vrátí seznam bounding boxů (x, y, w, h) kde pravděpodobně je text/číslo.
    """
    h, w = gray.shape

    # --- Předzpracování ---
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # --- MSER (Maximally Stable Extremal Regions) – skvělý pro text ---
    mser = cv2.MSER_create(
        5,                          # delta
        30,                         # min_area
        int(h * w * 0.005),         # max_area (max 0.5 % plochy)
        0.25,                       # max_variation
    )
    regions, _ = mser.detectRegions(blurred)

    # Převeď regiony na bounding boxy a filtruj
    candidates = []
    for region in regions:
        x0, y0, x1, y1 = (region[:, 0].min(), region[:, 1].min(),
                           region[:, 0].max(), region[:, 1].max())
        rw, rh = x1 - x0, y1 - y0
        if rw < 4 or rh < 4:
            continue
        aspect = rw / max(rh, 1)
        # Číslice jsou přibližně čtvercové nebo mírně vyšší
        if aspect < 0.1 or aspect > 4.0:
            continue
        candidates.append((x0, y0, rw, rh))

    # Také přidej detekce z adaptivního prahování
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 4)
    # Morfologie – spojí části číslic
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x0, y0, rw, rh = cv2.boundingRect(cnt)
        if rw < 4 or rh < 4:
            continue
        aspect = rw / max(rh, 1)
        if aspect < 0.1 or aspect > 4.0:
            continue
        area = cv2.contourArea(cnt)
        fill = area / max(rw * rh, 1)
        if fill < 0.1:
            continue
        candidates.append((x0, y0, rw, rh))

    return candidates


# ---------------------------------------------------------------------------
# 3.  GROUPOVÁNÍ KANDIDÁTŮ → BIB REGIONY
# ---------------------------------------------------------------------------

def group_text_candidates(candidates, img_h: int, img_w: int):
    """
    Seskupí textové kandidáty do skupin, které by mohly tvořit startovní číslo
    (= více číslic vedle sebe na podobné výškové úrovni).
    Vrátí seznam (x, y, w, h) potenciálních bib oblastí.
    """
    if not candidates:
        return []

    # Seřaď podle x
    boxes = sorted(set(candidates), key=lambda b: b[0])

    groups = []
    used = [False] * len(boxes)

    for i, (x0, y0, w0, h0) in enumerate(boxes):
        if used[i]:
            continue
        group = [i]
        cy0 = y0 + h0 / 2

        for j, (x1, y1, w1, h1) in enumerate(boxes):
            if i == j or used[j]:
                continue
            cy1 = y1 + h1 / 2
            # Podobná výška (středy do 0.6× výšky od sebe)
            if abs(cy0 - cy1) > max(h0, h1) * 0.6:
                continue
            # Podobná velikost (výška do 2×)
            if max(h0, h1) / max(min(h0, h1), 1) > 3.0:
                continue
            # Horizontální blízkost (mezera max 3× šířka znaku)
            gap = x1 - (x0 + w0)
            if -w0 < gap < max(w0, w1) * 3:
                group.append(j)

        if len(group) >= 2:  # aspoň 2 znaky vedle sebe
            # Bounding box celé skupiny
            xs = [boxes[k][0] for k in group]
            ys = [boxes[k][1] for k in group]
            ws = [boxes[k][2] for k in group]
            hs = [boxes[k][3] for k in group]
            gx = min(xs)
            gy = min(ys)
            gw = max(x + w for x, w in zip(xs, ws)) - gx
            gh = max(y + h for y, h in zip(ys, hs)) - gy

            # Přidej trochu padding
            pad_x = max(int(gw * 0.15), 4)
            pad_y = max(int(gh * 0.2), 4)
            gx = max(0, gx - pad_x)
            gy = max(0, gy - pad_y)
            gw = min(img_w - gx, gw + 2 * pad_x)
            gh = min(img_h - gy, gh + 2 * pad_y)

            groups.append((gx, gy, gw, gh))
            for k in group:
                used[k] = True

    # Deduplikace překrývajících se boxů (NMS)
    groups = non_max_suppression(groups)
    return groups


def non_max_suppression(boxes, overlap_thresh=0.5):
    """Odstraní překrývající se boxy, ponechá větší."""
    if not boxes:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    x1, y1, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x2, y2 = x1 + w, y1 + h
    areas = w * h
    order = areas.argsort()[::-1]
    keep = []
    suppressed = np.zeros(len(boxes), dtype=bool)

    for i in range(len(order)):
        idx = order[i]
        if suppressed[idx]:
            continue
        keep.append(idx)
        for j in range(i + 1, len(order)):
            jdx = order[j]
            if suppressed[jdx]:
                continue
            # IoU
            ix1 = max(x1[idx], x1[jdx])
            iy1 = max(y1[idx], y1[jdx])
            ix2 = min(x2[idx], x2[jdx])
            iy2 = min(y2[idx], y2[jdx])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = areas[idx] + areas[jdx] - inter
            if inter / max(union, 1) > overlap_thresh:
                suppressed[jdx] = True

    return [tuple(map(int, boxes[k])) for k in keep]


# ---------------------------------------------------------------------------
# 4.  OCR NA VÝŘEZU
# ---------------------------------------------------------------------------

def preprocess_for_ocr(roi: np.ndarray) -> np.ndarray:
    """
    Připraví výřez pro Tesseract – zvětší, normalizuje kontrast,
    prahuje, přidá bílý okraj.
    """
    # Zvětši na rozumnou výšku
    target_h = 64
    scale = target_h / max(roi.shape[0], 1)
    new_w = max(int(roi.shape[1] * scale), 1)
    resized = cv2.resize(roi, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    # CLAHE – vyrovnání histogramu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(resized)

    # Otsu prahování
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Okraj
    bordered = cv2.copyMakeBorder(binary, 10, 10, 10, 10,
                                  cv2.BORDER_CONSTANT, value=255)
    return bordered


def ocr_digits(roi_gray: np.ndarray) -> str:
    """
    Spustí Tesseract v módu pro číslice, vrátí string číslic.
    Pokud Tesseract není nainstalován, vrátí prázdný řetězec.
    """
    img = preprocess_for_ocr(roi_gray)

    config = (
        '--oem 3 '          # LSTM engine
        '--psm 7 '          # Single text line
        '-c tessedit_char_whitelist=0123456789'
    )
    try:
        text = pytesseract.image_to_string(img, config=config)
    except TesseractNotFoundError:
        raise TesseractNotFoundError(
            "Tesseract OCR nebyl nalezen. Nainstalujte jej:\n"
            "  Windows:  https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  macOS:    brew install tesseract\n"
            "  Linux:    sudo apt install tesseract-ocr"
        )
    # Ponech jen číslice
    digits = ''.join(c for c in text if c.isdigit())
    return digits


def deskew_roi(roi_gray: np.ndarray) -> np.ndarray:
    """
    Opraví drobné natočení výřezu pomocí momentů.
    """
    _, thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 5:
        return roi_gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 1:
        return roi_gray
    (h, w) = roi_gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(roi_gray, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


# ---------------------------------------------------------------------------
# 5.  HLAVNÍ FUNKCE
# ---------------------------------------------------------------------------

def detect_bibs(image_path: str, out_dir: str = None, debug: bool = False):
    """
    Zpracuje jeden obrázek a vrátí seznam detekovaných startovních čísel.

    Returns:
        list[int] – unikátní detekovaná čísla (seřazená)
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[CHYBA] Nelze načíst: {image_path}")
        return []

    # Zmenši velké fotky pro rychlost (zachová aspect ratio)
    max_dim = 1600
    h, w = img_bgr.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        h, w = img_bgr.shape[:2]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Příprava výstupního adresáře
    stem = Path(image_path).stem
    if out_dir:
        save_dir = Path(out_dir)
    else:
        save_dir = Path(image_path).parent / f"{stem}.out"
    save_dir.mkdir(parents=True, exist_ok=True)

    if debug:
        cv2.imwrite(str(save_dir / "0_input.jpg"), img_bgr)

    # --- Detekce kandidátů ---
    candidates = find_text_candidates(gray, debug=debug)

    if debug:
        dbg = img_bgr.copy()
        for (x, y, bw, bh) in candidates:
            cv2.rectangle(dbg, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
        cv2.imwrite(str(save_dir / "1_candidates.jpg"), dbg)

    # --- Groupování ---
    groups = group_text_candidates(candidates, h, w)

    if debug:
        dbg2 = img_bgr.copy()
        for (x, y, bw, bh) in groups:
            cv2.rectangle(dbg2, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv2.imwrite(str(save_dir / "2_groups.jpg"), dbg2)

    # --- OCR každé skupiny ---
    results = []
    bib_count = 0

    for (x, y, bw, bh) in groups:
        roi_gray = gray[y:y + bh, x:x + bw]
        if roi_gray.size == 0:
            continue

        # Deskew
        roi_deskewed = deskew_roi(roi_gray)

        # OCR
        number_str = ocr_digits(roi_deskewed)

        # Filtruj: startovní čísla mají typicky 2–5 číslic, >9
        if len(number_str) >= 2:
            try:
                num = int(number_str)
                if num >= 10:  # ignoruj jednociferná (nízká přesnost)
                    results.append(num)
                    # Ulož výřez
                    roi_color = img_bgr[y:y + bh, x:x + bw]
                    out_path = save_dir / f"bib-{bib_count:05d}-{num:04d}.png"
                    cv2.imwrite(str(out_path), roi_color)
                    bib_count += 1
            except ValueError:
                pass

    results = sorted(set(results))

    # Finální anotovaný obrázek
    final = img_bgr.copy()
    for (x, y, bw, bh) in groups:
        roi_gray = gray[y:y + bh, x:x + bw]
        number_str = ocr_digits(deskew_roi(roi_gray))
        if len(number_str) >= 2:
            try:
                num = int(number_str)
                if num >= 10:
                    cv2.rectangle(final, (x, y), (x + bw, y + bh), (0, 200, 0), 2)
                    cv2.putText(final, str(num), (x, y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
            except ValueError:
                pass
    cv2.imwrite(str(save_dir / "annotated.jpg"), final)

    return results


# ---------------------------------------------------------------------------
# 6.  CLI
# ---------------------------------------------------------------------------

def check_tesseract():
    """Ověří, že Tesseract je nainstalován a dostupný v PATH.
    Na Windows zkusí i běžné instalační cesty."""
    if shutil.which("tesseract") is not None:
        return

    # Na Windows zkusit běžné instalační cesty
    if sys.platform == "win32":
        common_paths = [
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                         "Tesseract-OCR", "tesseract.exe"),
            os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                         "Tesseract-OCR", "tesseract.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""),
                         "Programs", "Tesseract-OCR", "tesseract.exe"),
        ]
        for path in common_paths:
            if path and os.path.isfile(path):
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"[INFO] Tesseract nalezen: {path}")
                return

    print(
        "[CHYBA] Tesseract OCR nebyl nalezen.\n"
        "Bez Tesseractu nelze rozpoznávat čísla.\n\n"
        "Instalace:\n"
        "  Windows:  stáhněte z https://github.com/UB-Mannheim/tesseract/wiki\n"
        "            a přidejte instalační složku do systémové proměnné PATH\n"
        "  macOS:    brew install tesseract\n"
        "  Linux:    sudo apt install tesseract-ocr\n"
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Detekce startovních čísel ze závodních fotek",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("input", help="Cesta k obrázku nebo složce")
    parser.add_argument("--out", default=None, help="Výstupní složka (default: vedle vstupu)")
    parser.add_argument("--debug", action="store_true", help="Ulož mezikroky zpracování")
    args = parser.parse_args()

    check_tesseract()

    input_path = Path(args.input)
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    if input_path.is_dir():
        files = [f for f in sorted(input_path.iterdir())
                 if f.suffix.lower() in img_extensions]
        print(f"Zpracovávám {len(files)} obrázků ze složky {input_path}/")
        for f in files:
            numbers = detect_bibs(str(f), out_dir=args.out, debug=args.debug)
            print(f"{f.name}: {sorted(numbers) if numbers else '(nic nenalezeno)'}")
    elif input_path.is_file():
        numbers = detect_bibs(str(input_path), out_dir=args.out, debug=args.debug)
        if numbers:
            print(f"\nNalezená startovní čísla: {numbers}")
        else:
            print("\nŽádná startovní čísla nebyla nalezena.")
    else:
        print(f"[CHYBA] '{args.input}' neexistuje.")
        sys.exit(1)


if __name__ == "__main__":
    main()
