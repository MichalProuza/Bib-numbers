# Bib Numbers – detekce startovních čísel ze závodních fotek

Python reimplementace projektu [gheinrich/bibnumber](https://github.com/gheinrich/bibnumber), která nahrazuje původní pipeline OpenCV/Tesseract moderními deep-learningovými OCR enginy – EasyOCR a PaddleOCR.

---

## Požadavky

**Doporučená verze Pythonu: 3.10 nebo 3.11.**
PyTorch (EasyOCR) a PaddlePaddle nemusí podporovat Python 3.13.

---

## Instalace závislostí

```bash
# Základní zpracování obrazu
pip install opencv-python numpy

# OCR engine – nainstalujte alespoň jeden:
pip install easyocr                   # výchozí engine
pip install paddleocr paddlepaddle    # alternativa (CPU)
# pip install paddlepaddle-gpu        # alternativa (GPU)

# Volitelné: zápis klíčových slov do EXIF (Windows Průzkumník)
pip install piexif pillow
```

---

## Rychlý start

```bash
# Zpracování jednoho obrázku (výchozí engine: EasyOCR)
python3 bibnumber.py test_race.jpg

# Použití PaddleOCR a vlastní výstupní složky
python3 bibnumber.py test_race.jpg --engine paddleocr --out vysledky/

# Zpracování celé složky
python3 bibnumber.py slozka_s_fotkami/

# GUI aplikace
python3 app.py
```

---

## Jak to vyzkoušet

### 1. Nainstaluj závislosti

```bash
pip install opencv-python numpy easyocr
```

### 2. Spusť detekci na testovací fotce

V repozitáři je přiložená testovací fotka `test_race.jpg` se třemi běžci:

```bash
python3 bibnumber.py test_race.jpg
```

Při prvním spuštění EasyOCR stáhne modely (~100 MB). Poté uvidíš výstup podobný:

```
[INFO] Inicializuji EasyOCR [CPU] …
[INFO] EasyOCR připraven [CPU].
Nalezená startovní čísla: [38, 164, 775]
```

### 3. Zkontroluj výsledky

Spuštění s `--out` uloží anotovaný obrázek do zadané složky:

```bash
python3 bibnumber.py test_race.jpg --out vysledky/
```

Ve složce `vysledky/` najdeš `test_race.jpg` s zelenými rámečky a popisky kolem detekovaných čísel.

### 4. Vyzkoušej vlastní fotku

```bash
python3 bibnumber.py moje_fotka.jpg --out vysledky/
```

Pro nejlepší výsledky použij fotku, kde jsou startovní čísla čitelná pouhým okem (alespoň 1000×700 px, bez silného rozmazání).

### 5. Použití jako Python modul

```python
from bibnumber import detect_bibs

cisla = detect_bibs("test_race.jpg")
print(cisla)  # [38, 164, 775]

# Vlastní výstupní složka, alternativní engine
cisla = detect_bibs("foto.jpg", out_dir="vysledky/", engine="paddleocr")
```

---

## GUI aplikace (`app.py`)

```bash
python3 app.py
```

Grafická aplikace umožňuje:

- Výběr složky s fotkami přes dialog
- Volbu OCR enginu (EasyOCR / PaddleOCR) – nedostupné enginy jsou automaticky zakázány
- Průběžný výpis výsledků v terminálovém stylu
- Uložení anotovaných kopií do podsložky `_annotated/`
- Zápis detekovaných čísel jako klíčová slova do metadat JPEG:
  - **IPTC dataset 2:25** – čitelné v Lightroomu, Capture One, exiftool
  - **EXIF XPKeywords** (vyžaduje `piexif`) – čitelné v Windows Průzkumníku

---

## Architektura

```
bibnumber.py  ←  app.py   (hlavní GUI: zpracování složky + zápis IPTC metadat)
bibnumber.py  ←  gui.py   (legacy GUI: Tesseract-based, nahrazeno app.py)
```

`bibnumber.py` je jádro detekce; obě GUI z něj importují `detect_bibs()`. `gui.py` je ponechán z původní implementace, aktivní GUI je `app.py`.

### Detekční pipeline (`bibnumber.py`)

`detect_bibs(image_path, out_dir, debug, engine)` → `list[int]`

1. **Resize** – obrázky jsou zmenšeny na max. 1200 px (EasyOCR) nebo 1024 px (PaddleOCR) na nejdelší straně.
2. **OCR** – raw detekce z vybraného enginu. Oba jsou lazily inicializovány jako singleton; první volání stáhne modely (~100 MB pro EasyOCR).
3. **Filtrování** – každá detekce musí projít:
   - confidence ≥ 0.35
   - výška ≥ 12 px, aspect ratio 0.4–9.0
   - text je celý číselný (bez písmen)
   - 2–6 číslic
   - není opakující se vzor (např. „1111")
   - hodnota ≥ 10
4. **Vizuální validace** (`_looks_like_bib`) – ověří, zda region má dostatečný kontrast (≥ 35) a uniformní okolí (std < 65). Přeskočeno pro confidence ≥ 0.7.
5. **Výstup** – vrátí `sorted(set(results))`; volitelně uloží anotovaný JPEG do `out_dir`.

Systém je laděn na **vysokou přesnost**: raději číslo vynechá, než aby hlásil špatné.

### Kompatibilita PaddleOCR API

PaddleOCR změnil Python API mezi v2.x a v3.x. `_get_paddleocr_reader()` zkouší čtyři varianty konstruktoru postupně, aby transparentně podporoval obě verze.

---

## Klíčové parametry (ladění)

| Parametr | Místo v kódu | Efekt |
|----------|-------------|-------|
| `conf < 0.35` | `detect_bibs` | Minimální OCR confidence |
| `bh_px < 12` | `detect_bibs` | Minimální výška bibu v pixelech |
| `ratio 0.4–9.0` | `detect_bibs` | Ochrana aspect ratio |
| `conf < 0.7` přeskočí vizuální check | `detect_bibs` | Velmi jisté detekce obejdou `_looks_like_bib` |
| contrast `< 35` | `_looks_like_bib` | Min. rozdíl průměrů světlých/tmavých pixelů |
| bg std `< 65` | `_looks_like_bib` | Max. std pozadí (nižší = přísnější) |
| `max_dim` 1200/1024 | `detect_bibs` | Limit zmenšení dle enginu |

---

## Zpracování složky jako modul

```python
from pathlib import Path
from bibnumber import detect_bibs

for foto in sorted(Path("zavod_2026/").glob("*.jpg")):
    cisla = detect_bibs(str(foto), out_dir="vysledky/")
    print(f"{foto.name}: {cisla}")
```

---

## Závislosti

| Balíček | Verze | Účel |
|---------|-------|------|
| `opencv-python` | ≥ 4.5 | Zpracování obrazu, resize, anotace |
| `numpy` | ≥ 1.20 | Maticové operace |
| `easyocr` | ≥ 1.6 | Deep-learning OCR engine (výchozí) |
| `paddleocr` | ≥ 2.6 | Alternativní OCR engine |
| `paddlepaddle` | ≥ 2.4 | Backend pro PaddleOCR |
| `piexif` | libovolná | Zápis EXIF XPKeywords (volitelné) |
| `pillow` | ≥ 9.0 | Podpora piexif (volitelné) |

---

## Tipy pro lepší výsledky

- Fotky v rozlišení alespoň 1000×700 px
- Startovní číslo čitelné pouhým okem (bez silného pohybového rozmazání)
- Světlé číslo na tmavém pozadí i tmavé na světlém funguje stejně dobře
- EasyOCR je výchozí a doporučený engine; PaddleOCR je výrazně pomalejší bez GPU

**Co skript záměrně ignoruje:**
- Čísla kratší než 2 cifry (příliš vysoká chybovost)
- Překrývající se nebo silně rozmazané číslice
- Opakující se vzory („0000", „1111")

---

## Licence

Původní projekt [gheinrich/bibnumber](https://github.com/gheinrich/bibnumber) nemá explicitní licenci. Tato Python reimplementace vznikla jako nezávislé dílo pro osobní/výzkumné účely.
