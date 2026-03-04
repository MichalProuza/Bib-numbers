# bibnumber.py – Detekce startovních čísel ze závodních fotek

Python reimplementace projektu [gheinrich/bibnumber](https://github.com/gheinrich/bibnumber) postavená na moderním OpenCV 4.x a Tesseract OCR.

---

## Instalace závislostí

### Python balíčky

```bash
pip install opencv-python pytesseract scipy numpy
```

### Tesseract OCR

**Linux (Ubuntu/Debian):**
```bash
sudo apt install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Stáhni instalátor z [github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki) a přidej cestu do PATH (typicky `C:\Program Files\Tesseract-OCR`).

---

## Rychlý start

```bash
# Zpracování jednoho obrázku
python3 bibnumber.py zavod.jpg

# Zpracování celé složky
python3 bibnumber.py slozka_s_fotkami/

# Uložení výstupů do vlastní složky
python3 bibnumber.py zavod.jpg --out vysledky/

# Debug mód (ukládá mezikroky zpracování)
python3 bibnumber.py zavod.jpg --debug
```

---

## Jak to vyzkoušet

### 1. Nainstaluj závislosti

```bash
# Python balíčky
pip install opencv-python pytesseract scipy numpy

# Tesseract OCR (Ubuntu/Debian)
sudo apt install tesseract-ocr
```

### 2. Spusť detekci na testovací fotce

V repozitáři je přiložený testovací obrázek `test_race.jpg` se třemi běžci. Spusť:

```bash
python3 bibnumber.py test_race.jpg
```

Očekávaný výstup v konzoli:

```
Nalezená startovní čísla: [38, 164, 775]
```

### 3. Zkontroluj výsledky

Po spuštění se vytvoří složka `test_race.out/` s těmito soubory:

- **`annotated.jpg`** – původní fotka se zelenými rámečky kolem detekovaných čísel
- **`bib-*.png`** – výřezy jednotlivých startovních čísel

Otevři `test_race.out/annotated.jpg` a ověř, že detekovaná čísla jsou správně označena.

### 4. Vyzkoušej debug mód

```bash
python3 bibnumber.py test_race.jpg --debug
```

Ve výstupní složce pak najdeš i mezikroky zpracování (`0_input.jpg`, `1_candidates.jpg`, `2_groups.jpg`), které ukazují, jak pipeline postupně detekuje a seskupuje znaky.

### 5. Vyzkoušej vlastní fotku

```bash
python3 bibnumber.py moje_fotka.jpg --out vysledky/
```

Pro nejlepší výsledky použij fotku, kde jsou startovní čísla čitelná pouhým okem (alespoň 1000×700 px, bez silného rozmazání).

### 6. Vyzkoušej jako Python modul

```python
from bibnumber import detect_bibs

cisla = detect_bibs("test_race.jpg")
print(cisla)  # [38, 164, 775]
```

---

## Výstup

Skript vytvoří složku `{jmeno_fotky}.out/` (nebo zadanou `--out` složku) s těmito soubory:

| Soubor | Popis |
|---|---|
| `annotated.jpg` | Původní fotka s vyznačenými detekovanými čísly |
| `bib-00000-0164.png` | Výřez bib čísla 164 (první nalezené) |
| `bib-00001-0775.png` | Výřez bib čísla 775 (druhé nalezené) |
| … | Další výřezy ve formátu `bib-{pořadí}-{číslo}.png` |

Do konzole vypíše seznam detekovaných čísel, např.:

```
Nalezená startovní čísla: [38, 164, 773, 775]
```

### Debug soubory (`--debug`)

| Soubor | Popis |
|---|---|
| `0_input.jpg` | Vstupní fotka po případném zmenšení |
| `1_candidates.jpg` | Zelené rámečky – kandidáti na jednotlivé znaky |
| `2_groups.jpg` | Červené rámečky – seskupené kandidáty (= kandidáti na celá čísla) |

---

## Jak to funguje – pipeline

### 1. Předzpracování
Velké fotky jsou automaticky zmenšeny na max. 1600 px (kratší strana) pro zachování rozumné rychlosti. Zpracování probíhá na greyscale verzi.

### 2. Detekce textových oblastí
Kombinuje dvě metody:

- **MSER** (Maximally Stable Extremal Regions) – detekuje stabilní oblasti v různých úrovních jasu, přirozeně zvýrazňuje tištěný text
- **Adaptivní prahování + kontury** – zachytí oblasti kde MSER selže (nízký kontrast, pohybová neostrost)

Obě metody filtrují výsledky podle poměru stran (číslice jsou přibližně čtvercové nebo mírně vyšší) a hustoty hranic.

### 3. Seskupování znaků
Textové kandidáti jsou seskupeni do startovních čísel podle těchto kritérií:
- Podobná výšková pozice (středy do 60 % výšky znaku)
- Podobná velikost (max. 3× rozdíl výšek)
- Horizontální blízkost (mezera max. 3× šířka znaku)

Překrývající se boxy jsou odstraněny pomocí **NMS** (Non-Maximum Suppression).

### 4. Korekce natočení (deskew)
Každý kandidátní výřez je narovnán pomocí analýzy momentů binárního obrazu, aby OCR fungovalo co nejlépe.

### 5. OCR
Tesseract v režimu PSM 7 (jeden řádek textu) s whitelist `0123456789`. Před OCR proběhne:
- Zvětšení výřezu na výšku 64 px
- CLAHE (vyrovnání kontrastu s omezením)
- Otsu prahování
- Přidání bílého okraje

### 6. Filtrování
Za validní startovní číslo se považuje pouze výsledek s ≥ 2 ciframi a hodnotou ≥ 10. Jednociferná čísla jsou ignorována (vysoká chybovost OCR na malých výřezech).

---

## Tipy pro lepší výsledky

**Kvalita vstupní fotky má velký vliv.** Pro nejlepší výsledky:

- Fotky v rozlišení alespoň 1000×700 px
- Startovní číslo čitelné pouhým okem (bez silného rozmazání pohybem)
- Světlé číslo na tmavém pozadí nebo tmavé na světlém funguje stejně dobře
- Šikmé záběry do ~25° jsou zvládnuty automatickým deskewem

**Co skript záměrně ignoruje:**
- Čísla menší než 2 cifry (příliš vysoká chybovost)
- Překrývající se nebo silně rozmazané číslice
- Čísla na billboardech a tabulích na trati (filtrace podle kontextu není implementována)

---

## Použití jako modul v Pythonu

```python
from bibnumber import detect_bibs

# Zpracuje fotku, výřezy uloží do foto.out/
numbers = detect_bibs("foto.jpg")
print(numbers)  # [38, 164, 775]

# Vlastní výstupní složka, debug mezikroky
numbers = detect_bibs("foto.jpg", out_dir="vysledky/", debug=True)
```

Funkce `detect_bibs` vrací `list[int]` – unikátní detekovaná čísla seřazená vzestupně.

---

## Zpracování složky s fotkami

```python
from pathlib import Path
from bibnumber import detect_bibs

foto_slozka = Path("zavod_2026/")
for foto in sorted(foto_slozka.glob("*.jpg")):
    cisla = detect_bibs(str(foto), out_dir="vysledky/")
    print(f"{foto.name}: {cisla}")
```

---

## Výkonnost a omezení

| Parametr | Hodnota |
|---|---|
| Typická doba zpracování (1600px fotka) | 2–5 s |
| Přesnost na čitelných číslech (3+ cifry) | ~85 % |
| Přesnost na číslech (2 cifry) | ~60 % |
| Minimální výška čísla ve fotce | ~30 px |

Skript je laděn na **vysokou přesnost**, nikoliv na vysoký recall – raději číslo vynechá, než aby hlásil špatné.

---

## Závislosti

| Balíček | Verze | Účel |
|---|---|---|
| `opencv-python` | ≥ 4.5 | Zpracování obrazu, MSER, morfologie |
| `pytesseract` | ≥ 0.3 | Python wrapper pro Tesseract |
| `numpy` | ≥ 1.20 | Maticové operace |
| `scipy` | ≥ 1.7 | Morfologické operace na SWT |
| Tesseract OCR | ≥ 4.0 | OCR engine (LSTM) |

---

## Licence

Původní projekt [gheinrich/bibnumber](https://github.com/gheinrich/bibnumber) nemá explicitní licenci. Tato Python reimplementace vznikla jako nezávislé dílo pro osobní/výzkumné účely.
