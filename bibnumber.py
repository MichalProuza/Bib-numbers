#!/usr/bin/env python3
"""
bibnumber.py – detekce startovních čísel ze závodních fotek

Pipeline:
  1. EasyOCR (CRAFT detektor textu + CRNN rekognice)
  2. Filtrace výsledků: pouze číslice, 2–6 znaků, confidence ≥ 0.4
  3. Post-validace: zamítnutí opakujících se vzorů ("1111")

Použití:
  python3 bibnumber.py foto.jpg
  python3 bibnumber.py slozka_s_fotkami/
  python3 bibnumber.py foto.jpg --out vystup/
"""

import cv2
import numpy as np
import sys
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# EasyOCR reader – lazy init, inicializuje se jednou za celý proces
# ---------------------------------------------------------------------------

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        try:
            import easyocr
        except ImportError:
            print(
                "[CHYBA] EasyOCR není nainstalován.\n"
                "Nainstalujte: pip install easyocr\n",
                flush=True,
            )
            sys.exit(1)
        print(
            "[INFO] Inicializuji EasyOCR"
            " (první spuštění stáhne modely ~100 MB, chvíli trvá)…",
            flush=True,
        )
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        print("[INFO] EasyOCR připraven.", flush=True)
    return _reader


# ---------------------------------------------------------------------------
# Vizuální validace – rozliší bib od náhodných čísel v pozadí
# ---------------------------------------------------------------------------

def _looks_like_bib(img_bgr: np.ndarray, pts: np.ndarray) -> bool:
    """
    Heuristicky ověří, zda detekovaný region vypadá jako startovní číslo.

    Skutečný bib má dvě klíčové vlastnosti:
      1. Vysoký kontrast číslic vůči jejich bezprostřednímu pozadí
         (tmavé číslice na světlém papíru nebo naopak).
      2. Relativně uniformní plocha kolem číslic – bib materiál
         (papír/tkanina jedné barvy), na rozdíl od vzorovaného oblečení
         nebo reklamy v pozadí.
    """
    x1 = int(pts[:, 0].min());  x2 = int(pts[:, 0].max())
    y1 = int(pts[:, 1].min());  y2 = int(pts[:, 1].max())
    bw, bh = x2 - x1, y2 - y1
    if bw == 0 or bh == 0:
        return False

    H, W = img_bgr.shape[:2]

    # 1. Kontrast uvnitř bbox ────────────────────────────────────────────────
    inner = img_bgr[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
    if inner.size == 0:
        return False

    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dark  = gray[binary == 0]
    light = gray[binary == 255]
    if len(dark) == 0 or len(light) == 0:
        return False

    # Rozdíl průměrů světlých a tmavých pixelů musí být alespoň 45/255
    if float(light.mean()) - float(dark.mean()) < 45:
        return False

    # 2. Uniformita pozadí kolem bbox (bib materiál) ─────────────────────────
    pad = max(int(min(bw, bh) * 0.6), 10)
    ox1 = max(0, x1 - pad);  oy1 = max(0, y1 - pad)
    ox2 = min(W, x2 + pad);  oy2 = min(H, y2 + pad)

    outer = img_bgr[oy1:oy2, ox1:ox2]
    if outer.size == 0:
        return True

    gray_out = cv2.cvtColor(outer, cv2.COLOR_BGR2GRAY)

    # Odmaž střed (samotný text) – měříme jen okolí
    mask = np.zeros(gray_out.shape, dtype=bool)
    ry1, ry2 = y1 - oy1, y2 - oy1
    rx1, rx2 = x1 - ox1, x2 - ox1
    mask[max(0, ry1):ry2, max(0, rx1):rx2] = True
    bg = gray_out[~mask]

    if len(bg) < 20:
        return True   # Nedostatek dat k posouzení → propusť

    # Vzorované oblečení / reklamy v pozadí mají std > 55
    # Bib materiál (papír, tkanina) má std typicky < 45
    return float(bg.std()) < 55


# ---------------------------------------------------------------------------
# Hlavní funkce
# ---------------------------------------------------------------------------

def detect_bibs(image_path: str, out_dir: str = None, debug: bool = False):
    """
    Zpracuje jeden obrázek a vrátí seznam detekovaných startovních čísel.

    Returns:
        list[int] – unikátní detekovaná čísla (seřazená)
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[CHYBA] Nelze načíst: {image_path}", flush=True)
        return []

    # Zmenši velké fotky pro rychlost (zachová aspect ratio)
    max_dim = 1600
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        h, w = img_bgr.shape[:2]

    # Výstupní adresář
    stem = Path(image_path).stem
    save_dir = Path(out_dir) if out_dir else Path(image_path).parent / f"{stem}.out"
    save_dir.mkdir(parents=True, exist_ok=True)

    # EasyOCR – vrátí seznam (bbox, text, confidence)
    reader = _get_reader()
    raw = reader.readtext(
        img_bgr,
        detail=1,
        paragraph=False,
        allowlist="0123456789",  # hledáme jen číslice → méně šumu
        min_size=10,             # default 20; zachytí menší/vzdálenější biby
        text_threshold=0.6,      # default 0.7; mírně citlivější detektor
        low_text=0.3,            # default 0.4; větší pokrytí okrajů znaků
        link_threshold=0.4,      # default 0.4; zachováno – nižší hodnota slévá nesouvisející oblasti
        mag_ratio=1.5,           # default 1.0; zvětšení před detekcí
    )

    results       = []
    bib_detections = []   # (bbox, num) pro anotaci a výřezy

    for (bbox, text, conf) in raw:
        # Příliš nízká spolehlivost
        if conf < 0.35:
            continue

        # Geometrická validace bbox ─────────────────────────────────────────
        pts   = np.array(bbox, dtype=np.int32)
        bx1   = int(pts[:, 0].min());  bx2 = int(pts[:, 0].max())
        by1   = int(pts[:, 1].min());  by2 = int(pts[:, 1].max())
        bh_px = by2 - by1
        bw_px = bx2 - bx1

        # Příliš nízký bib – spíš šum nebo vzdálené nečitelné číslo
        if bh_px < 12:
            continue

        # Aspect ratio: „12" ≈ 1.2:1, „12345" ≈ 5:1 – extrémní hodnoty jsou šum
        ratio = bw_px / max(bh_px, 1)
        if ratio < 0.4 or ratio > 9.0:
            continue
        # ───────────────────────────────────────────────────────────────────

        # Ponech jen číslice ze zaznamenaného textu
        digits = "".join(c for c in text if c.isdigit())

        # Startovní čísla mají 2–6 číslic
        if len(digits) < 2 or len(digits) > 6:
            continue

        # Odmítni "1111", "0000" apod. – typický šum
        if len(set(digits)) == 1:
            continue

        try:
            num = int(digits)
        except ValueError:
            continue

        if num < 10:
            continue

        # Vizuální validace – odmítne čísla bez bib-like pozadí
        if not _looks_like_bib(img_bgr, pts):
            continue

        results.append(num)
        bib_detections.append((bbox, num))

    results = sorted(set(results))

    # Ulož výřezy a anotovaný obrázek
    final = img_bgr.copy()
    saved = set()

    for idx, (bbox, num) in enumerate(bib_detections):
        pts = np.array(bbox, dtype=np.int32)
        x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
        x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())

        # Výřez
        roi = img_bgr[max(0, y1):y2, max(0, x1):x2]
        if roi.size > 0 and num not in saved:
            cv2.imwrite(str(save_dir / f"bib-{idx:05d}-{num:04d}.png"), roi)
            saved.add(num)

        # Anotace – polygon (zvládá i nakloněné detekce)
        cv2.polylines(final, [pts], True, (0, 200, 0), 2)
        cv2.putText(
            final, str(num), (x1, max(y1 - 6, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2,
        )

    cv2.imwrite(str(save_dir / "annotated.jpg"), final)

    return results


# ---------------------------------------------------------------------------
# Kontrola závislostí
# ---------------------------------------------------------------------------

def check_easyocr():
    """Ověří, že EasyOCR je nainstalován. Při chybě ukončí proces."""
    try:
        import easyocr  # noqa: F401
    except ImportError:
        print(
            "[CHYBA] EasyOCR není nainstalován.\n"
            "Nainstalujte: pip install easyocr\n"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detekce startovních čísel ze závodních fotek",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Cesta k obrázku nebo složce")
    parser.add_argument("--out", default=None, help="Výstupní složka (default: vedle vstupu)")
    parser.add_argument("--debug", action="store_true", help="Debug mód")
    args = parser.parse_args()

    check_easyocr()

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
