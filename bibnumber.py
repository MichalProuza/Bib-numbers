#!/usr/bin/env python3
"""
bibnumber.py – detekce startovních čísel ze závodních fotek

Pipeline:
  1. OCR engine (EasyOCR nebo PaddleOCR)
  2. Filtrace výsledků: pouze číslice, 2–6 znaků, confidence ≥ 0.35
  3. Post-validace: zamítnutí opakujících se vzorů ("1111")
  4. Vizuální validace: kontrast + uniformita okolí (_looks_like_bib)

Použití:
  python3 bibnumber.py foto.jpg
  python3 bibnumber.py slozka_s_fotkami/
  python3 bibnumber.py foto.jpg --out vystup/ --engine paddleocr
"""

import cv2
import numpy as np
import sys
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# OCR readers – lazy init, každý se inicializuje jednou za celý proces
# ---------------------------------------------------------------------------

_easyocr_reader  = None
_paddleocr_reader = None


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
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
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        print("[INFO] EasyOCR připraven.", flush=True)
    return _easyocr_reader


def _get_paddleocr_reader():
    global _paddleocr_reader
    if _paddleocr_reader is None:
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            print(
                "[CHYBA] PaddleOCR není nainstalován.\n"
                "Nainstalujte: pip install paddleocr\n",
                flush=True,
            )
            sys.exit(1)
        print(
            "[INFO] Inicializuji PaddleOCR"
            " (první spuštění stáhne modely, chvíli trvá)…",
            flush=True,
        )
        # Potlač verbose výstup přes logging (show_log byl odstraněn v 3.x)
        import logging
        logging.getLogger("ppocr").setLevel(logging.ERROR)

        # Zkus varianty API postupně (3.x → 2.x → minimální)
        for kwargs in [
            {"lang": "en", "device": "cpu"},   # PaddleOCR 3.x
            {"lang": "en", "use_gpu": False, "show_log": False},  # 2.x
            {"lang": "en"},                     # absolutní záložka
        ]:
            try:
                _paddleocr_reader = PaddleOCR(**kwargs)
                break
            except Exception:
                continue
        else:
            print("[CHYBA] Nepodařilo se inicializovat PaddleOCR.", flush=True)
            sys.exit(1)
        print("[INFO] PaddleOCR připraven.", flush=True)
    return _paddleocr_reader


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

    # Rozdíl průměrů světlých a tmavých pixelů musí být alespoň 35/255
    # (původně 45 – příliš striktní pro vybledlé/mírně kontrastní biby)
    if float(light.mean()) - float(dark.mean()) < 35:
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

    # Uniformita okolí: std < 65 = pravděpodobně bib materiál
    # (vzorované oblečení / reklamy mívají std > 80)
    return float(bg.std()) < 65


# ---------------------------------------------------------------------------
# Hlavní funkce
# ---------------------------------------------------------------------------

def detect_bibs(image_path: str, out_dir: str = None, debug: bool = False,
                engine: str = "easyocr"):
    """
    Zpracuje jeden obrázek a vrátí seznam detekovaných startovních čísel.

    Args:
        engine: "easyocr" (výchozí) nebo "paddleocr"

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

    # OCR – výstup normalizujeme na jednotný formát [(bbox, text, conf), …]
    # kde bbox = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    if engine == "paddleocr":
        ocr = _get_paddleocr_reader()
        paddle_result = ocr.ocr(img_bgr, cls=True)
        raw = []
        if paddle_result and paddle_result[0]:
            for line in paddle_result[0]:
                box, (text, conf) = line
                raw.append((box, text, float(conf)))
    else:
        reader = _get_easyocr_reader()
        raw = reader.readtext(
            img_bgr,
            detail=1,
            paragraph=False,
            min_size=10,        # default 20; zachytí menší/vzdálenější biby
            text_threshold=0.6, # default 0.7; mírně citlivější detektor
            low_text=0.3,       # default 0.4; větší pokrytí okrajů znaků
            link_threshold=0.4, # default 0.4
            mag_ratio=1.5,      # default 1.0; zvětšení před detekcí
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

        # Přečtený text musí být CELÝ číselný (ignorujeme mezery).
        # Bez allowlistu EasyOCR přečte "SPORT" jako "SPORT" → zamítneme.
        # S allowlistem by ho přečetl jako "5P0R7" → extrahoval by "507" → false positive.
        digits = "".join(c for c in text if not c.isspace())
        if not digits.isdigit():
            continue

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

        # Vizuální validace – odmítne čísla bez bib-like pozadí.
        # Pro velmi jisté detekce (conf ≥ 0.7) validaci přeskočíme – čistě
        # číselný text s takovou spolehlivostí je prakticky vždy reálný bib
        # a heuristika občas filtrovala i zjevná čísla.
        if conf < 0.7 and not _looks_like_bib(img_bgr, pts):
            continue

        results.append(num)
        bib_detections.append((bbox, num))

    results = sorted(set(results))

    # Anotovaný obrázek – uloží se do out_dir s původním názvem fotky
    if out_dir and bib_detections:
        final = img_bgr.copy()
        for (bbox, num) in bib_detections:
            pts_a = np.array(bbox, dtype=np.int32)
            x1 = int(pts_a[:, 0].min())
            y1 = int(pts_a[:, 1].min())

            # Polygon kolem nalezeného čísla
            cv2.polylines(final, [pts_a], True, (0, 200, 0), 3)

            # Label: zelený rámeček + bílý text → čitelné na jakémkoli pozadí
            label = str(num)
            font  = cv2.FONT_HERSHEY_SIMPLEX
            (lw, lh), bl = cv2.getTextSize(label, font, 0.9, 2)
            ly = max(y1 - 6, lh + 6)
            cv2.rectangle(
                final,
                (x1 - 2,      ly - lh - 4),
                (x1 + lw + 4, ly + bl),
                (0, 180, 0), cv2.FILLED,
            )
            cv2.putText(final, label, (x1 + 1, ly - 1), font, 0.9, (255, 255, 255), 2)

        out_path = Path(out_dir) / Path(image_path).name
        cv2.imwrite(str(out_path), final)

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


def check_paddleocr():
    """Ověří, že PaddleOCR je nainstalován. Při chybě ukončí proces."""
    try:
        from paddleocr import PaddleOCR  # noqa: F401
    except ImportError:
        print(
            "[CHYBA] PaddleOCR není nainstalován.\n"
            "Nainstalujte: pip install paddleocr\n"
        )
        sys.exit(1)


def is_easyocr_available() -> bool:
    try:
        import easyocr  # noqa: F401
        return True
    except ImportError:
        return False


def is_paddleocr_available() -> bool:
    try:
        from paddleocr import PaddleOCR  # noqa: F401
        return True
    except ImportError:
        return False


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
    parser.add_argument("--engine", choices=["easyocr", "paddleocr"],
                        default="easyocr", help="OCR engine (default: easyocr)")
    parser.add_argument("--debug", action="store_true", help="Debug mód")
    args = parser.parse_args()

    if args.engine == "paddleocr":
        check_paddleocr()
    else:
        check_easyocr()

    input_path = Path(args.input)
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    if input_path.is_dir():
        files = [f for f in sorted(input_path.iterdir())
                 if f.suffix.lower() in img_extensions]
        print(f"Zpracovávám {len(files)} obrázků ze složky {input_path}/")
        for f in files:
            numbers = detect_bibs(str(f), out_dir=args.out, debug=args.debug,
                                  engine=args.engine)
            print(f"{f.name}: {sorted(numbers) if numbers else '(nic nenalezeno)'}")
    elif input_path.is_file():
        numbers = detect_bibs(str(input_path), out_dir=args.out, debug=args.debug,
                              engine=args.engine)
        if numbers:
            print(f"\nNalezená startovní čísla: {numbers}")
        else:
            print("\nŽádná startovní čísla nebyla nalezena.")
    else:
        print(f"[CHYBA] '{args.input}' neexistuje.")
        sys.exit(1)


if __name__ == "__main__":
    main()
