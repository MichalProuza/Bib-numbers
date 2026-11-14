# Bib Numbers – detekce startovních čísel ze závodních fotek

Python reimplementace projektu [gheinrich/bibnumber](https://github.com/gheinrich/bibnumber), která nahrazuje původní pipeline OpenCV/Tesseract moderními deep-learningovými OCR enginy – EasyOCR a PaddleOCR.

> `README_1.md` je záměrně ponechaná dokumentace **staré** implementace postavené na Tesseractu. Příkazy, přepínače (`--debug` s mezivýsledky, složka `*.out/`) ani pipeline v ní popsané pro současný kód neplatí. Aktuální stav popisuje tento soubor.

---

## Požadavky

**Doporučená verze Pythonu: 3.10 nebo 3.11.**
PyTorch (EasyOCR) a PaddlePaddle nemusí podporovat Python 3.13.

GUI (`app.py`) potřebuje `tkinter`. Ten je součástí standardní knihovny Pythonu, na některých linuxových distribucích se ale instaluje zvlášť (např. balíček `python3-tk`).

---

## Instalace závislostí

Repozitář neobsahuje `requirements.txt` ani jiný instalační soubor – balíčky se instalují ručně a jejich verze nejsou nikde zafixované.

```bash
# Základní zpracování obrazu
pip install opencv-python numpy

# OCR engine – nainstalujte alespoň jeden:
pip install easyocr                   # výchozí engine
pip install paddleocr paddlepaddle    # alternativa (CPU)
# pip install paddlepaddle-gpu        # alternativa (GPU)

# Volitelné: zápis klíčových slov do EXIF XPKeywords (Windows Průzkumník)
pip install piexif
```

---

## Rychlý start

```bash
# Zpracování jednoho obrázku (výchozí engine: EasyOCR)
python3 bibnumber.py test_race.jpg

# Použití PaddleOCR a uložení anotované kopie (složka musí předem existovat)
mkdir -p vysledky
python3 bibnumber.py test_race.jpg --engine paddleocr --out vysledky/

# Zpracování celé složky
python3 bibnumber.py slozka_s_fotkami/

# GUI aplikace
python3 app.py
```

### Přepínače CLI

| Argument | Význam |
|----------|--------|
| `input` | Cesta k obrázku nebo ke složce. Složka se neprochází rekurzivně; berou se soubory s příponou `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`. |
| `--out SLOŽKA` | Složka pro anotované kopie. Bez `--out` se žádný obrázek neukládá (text nápovědy „default: vedle vstupu“ neodpovídá chování kódu). |
| `--engine {easyocr,paddleocr}` | OCR engine, výchozí `easyocr`. |
| `--debug` | Přepínač existuje, ale v současném kódu nemá žádný účinek. |

Pozor na `--out`:

- **Složka musí existovat.** Skript ji nevytváří; když chybí, anotovaná kopie se tiše neuloží a skript neohlásí chybu.
- **Nepoužívej jako `--out` složku se zdrojovými fotkami.** Anotovaná kopie se ukládá pod stejným názvem jako vstup, takže by originál přepsala (zmenšenou verzí s rámečky).

---

## Jak to vyzkoušet

### 1. Nainstaluj závislosti

```bash
pip install opencv-python numpy easyocr
```

### 2. Spusť detekci na testovacím obrázku

V repozitáři je přiložený testovací obrázek `test_race.jpg` (900×600 px). Není to skutečná fotka ze závodu, ale syntetická scéna: tři barevné obdélníky místo běžců, každý s bílou cedulkou s číslem (164, 775, 38).

```bash
python3 bibnumber.py test_race.jpg
```

Při prvním spuštění EasyOCR stáhne modely (~100 MB). Poté uvidíš výstup podobný:

```
[INFO] Inicializuji EasyOCR [CPU] (první spuštění stáhne modely ~100 MB, chvíli trvá)…
[INFO] EasyOCR připraven [CPU].

Nalezená startovní čísla: [38, 164, 775]
```

Když skript nic nenajde, vypíše `Žádná startovní čísla nebyla nalezena.`

### 3. Zkontroluj výsledky

Spuštění s `--out` uloží anotovaný obrázek do zadané (existující) složky:

```bash
mkdir -p vysledky
python3 bibnumber.py test_race.jpg --out vysledky/
```

Ve složce `vysledky/` najdeš `test_race.jpg` se zelenými rámečky kolem detekovaných čísel a zelenými štítky s bílým textem. Kopie se ukládá jen tehdy, když bylo nalezeno alespoň jedno číslo, a má rozlišení po zmenšení (viz pipeline), ne rozlišení originálu.

Soubor `annotated.jpg` v kořeni repozitáře je ukázkový výstup staré Tesseract implementace (viz `README_1.md`); současný kód ho nevytváří ani nepoužívá.

### 4. Vyzkoušej vlastní fotku

```bash
python3 bibnumber.py moje_fotka.jpg --out vysledky/
```

Pro nejlepší výsledky použij fotku, kde jsou startovní čísla čitelná pouhým okem (alespoň 1000×700 px, bez silného rozmazání).

### 5. Použití jako Python modul

```python
from bibnumber import detect_bibs

cisla = detect_bibs("test_race.jpg")
print(cisla)  # seznam čísel, např. [38, 164, 775]

# Vlastní výstupní složka (musí existovat), alternativní engine
cisla = detect_bibs("foto.jpg", out_dir="vysledky/", engine="paddleocr")
```

Chování, se kterým je dobré počítat:

- Obrázek, který nejde načíst, vypíše `[CHYBA] Nelze načíst: …` a funkce vrátí `[]` (výjimku nevyhodí).
- Když chybí vybraný OCR engine nebo selže jeho inicializace, funkce vypíše chybu a zavolá `sys.exit(1)` – při použití jako knihovna tedy vyhodí `SystemExit`, ne běžnou výjimku.
- Parametr `debug` je v signatuře, ale nic nedělá.
- Jiná hodnota `engine` než `"paddleocr"` se zpracuje jako EasyOCR.

---

## GUI aplikace (`app.py`)

```bash
python3 app.py
```

Grafická aplikace umožňuje:

- Výběr složky s fotkami přes dialog (zpracují se soubory `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff` přímo ve složce, bez podsložek)
- Volbu OCR enginu (EasyOCR / PaddleOCR) – nedostupné enginy jsou automaticky zakázány; pokud nefunguje ani jeden, aplikace zobrazí chybové okno
- Průběžný výpis výsledků v terminálovém stylu (do okna i na standardní výstup) a ukazatel průběhu
- Tlačítko **Zastavit** – zpracování skončí po dokončení právě rozpracované fotky
- Uložení anotovaných kopií do podsložky `_annotated/` (vytvoří se automaticky)
- Zápis detekovaných čísel jako klíčová slova do metadat JPEG:
  - **IPTC dataset 2:25** – zapisuje se vždy, bez externí knihovny
  - **EXIF XPKeywords** (vyžaduje `piexif`) – pro Průzkumník Windows

Zpracování běží ve vlákně na pozadí, takže okno během práce nezamrzne.

### Zápis metadat – co je potřeba vědět

Zápis metadat **mění originální soubory přímo na místě, bez zálohy**. Doporučujeme pracovat nad kopií složky.

- Metadata se zapisují jen do `.jpg` / `.jpeg` a jen u fotek, na kterých bylo nalezeno aspoň jedno číslo. Ostatní formáty se přeskočí.
- Všechna čísla z jedné fotky se zapíší jako **jeden** záznam 2:25 s čísly oddělenými čárkou (např. `38, 164, 775`), ne jako samostatné klíčové slovo pro každé číslo. Stejný řetězec jde do XPKeywords.
- Záznam je oříznut na 255 bajtů; u fotky s velkým množstvím čísel se tak seznam usekne (i uprostřed čísla).
- Před zápisem se z JPEG **odstraní všechny existující segmenty APP13**. Dosavadní IPTC údaje (popisek, copyright, autor, dřívější klíčová slova) a další Photoshop bloky v souboru tím zaniknou.
- Nový segment APP13 se vkládá hned za značku SOI, tedy před případný segment APP0 (JFIF).
- Při opakovaném spuštění se klíčová slova nahradí aktuálním výsledkem detekce.

---

## Architektura

```
bibnumber.py  ←  app.py   (hlavní GUI: zpracování složky + zápis IPTC metadat)
bibnumber.py  ←  gui.py   (legacy GUI z doby Tesseract implementace, nahrazeno app.py)
```

`bibnumber.py` je jádro detekce; obě GUI z něj importují `detect_bibs()`. Aktivní GUI je `app.py`.

`gui.py` je pozůstatek původní implementace: detekci sice volá přes současné `detect_bibs()` (tedy EasyOCR), ale při startu importuje `pytesseract` a kontroluje přítomnost Tesseractu. Bez nainstalovaného balíčku `pytesseract`, který jinak projekt nepotřebuje, se vůbec nespustí. Neumí volbu enginu, anotované kopie ani zápis metadat.

### Detekční pipeline (`bibnumber.py`)

`detect_bibs(image_path, out_dir=None, debug=False, engine="easyocr")` → `list[int]`

1. **Resize** – obrázky jsou zmenšeny na max. 1200 px (EasyOCR) nebo 1024 px (PaddleOCR) na nejdelší straně. Menší obrázky se nezvětšují.
2. **OCR** – raw detekce z vybraného enginu. Oba jsou lazily inicializovány jako singleton na úrovni modulu (`_easyocr_reader`, `_paddleocr_reader`); první volání stáhne modely (~100 MB pro EasyOCR). EasyOCR běží s anglickým modelem a parametry `min_size=10`, `text_threshold=0.6`, `low_text=0.3`, `link_threshold=0.4`, `mag_ratio=1.2`, `batch_size=4`. GPU (CUDA) se detekuje automaticky; na CPU se u EasyOCR zapíná kvantizace.
3. **Filtrování** – každá detekce musí projít:
   - confidence ≥ 0.35
   - výška ≥ 12 px, aspect ratio 0.4–9.0
   - text je po odstranění mezer celý číselný (bez písmen); „12 34“ se tedy bere jako 1234
   - 2–6 číslic
   - není opakující se vzor (např. „1111“)
   - hodnota ≥ 10
4. **Vizuální validace** (`_looks_like_bib`) – ověří, zda region má dostatečný kontrast (rozdíl průměrů světlých a tmavých pixelů po Otsu prahování ≥ 35) a uniformní okolí (std < 65). Přeskočeno pro confidence ≥ 0.7.
5. **Výstup** – vrátí `sorted(set(results))`; pokud je zadán `out_dir` a něco se našlo, uloží anotovaný obrázek pod původním názvem souboru do `out_dir`.

Čísla se vrací jako `int`, takže úvodní nuly se ztrácejí („0123“ → 123, „007“ se zahodí jako hodnota < 10).

Systém je laděn na **vysokou přesnost**: raději číslo vynechá, než aby hlásil špatné.

### Kompatibilita PaddleOCR API

PaddleOCR změnil Python API mezi v2.x a v3.x. `_get_paddleocr_reader()` zkouší postupně čtyři varianty konstruktoru – bez parametrů, `lang` + `device`, `lang` + `use_gpu` + `show_log`, jen `lang` – a na další přechází, jen když chybová zpráva vypadá jako neznámý argument; jiná chyba inicializaci ukončí. Volání `ocr()` se zkouší s `cls=True` a při `TypeError` bez něj. Před importem se nastavují proměnné `FLAGS_use_mkldnn=0` a `PADDLE_DISABLE_ONEDNN=1` (vypnutí oneDNN), pokud už nejsou nastavené.

V GUI se dostupnost EasyOCR kontroluje dřív než PaddleOCR – na Windows mohou knihovny PaddlePaddle kolidovat s PyTorch, pokud se načtou v opačném pořadí.

---

## Klíčové parametry (ladění)

Prahy jsou napsané přímo v kódu, nejsou dostupné přes CLI ani konfiguraci.

| Parametr | Místo v kódu | Efekt |
|----------|-------------|-------|
| `conf < 0.35` | `detect_bibs` | Minimální OCR confidence |
| `bh_px < 12` | `detect_bibs` | Minimální výška bibu v pixelech |
| `ratio 0.4–9.0` | `detect_bibs` | Ochrana aspect ratio |
| `conf < 0.7` → vizuální check | `detect_bibs` | Detekce s confidence ≥ 0.7 obejdou `_looks_like_bib` |
| contrast `< 35` | `_looks_like_bib` | Min. rozdíl průměrů světlých/tmavých pixelů |
| bg std `< 65` | `_looks_like_bib` | Max. std pozadí (nižší = přísnější) |
| `max_dim` 1200/1024 | `detect_bibs` | Limit zmenšení dle enginu |

---

## Zpracování složky jako modul

```python
from pathlib import Path
from bibnumber import detect_bibs

Path("vysledky").mkdir(exist_ok=True)
for foto in sorted(Path("zavod_2026/").glob("*.jpg")):
    cisla = detect_bibs(str(foto), out_dir="vysledky/")
    print(f"{foto.name}: {cisla}")
```

---

## Závislosti

Minimální verze nejsou v repozitáři nikde určené ani otestované.

| Balíček | Účel |
|---------|------|
| `opencv-python` | Načtení obrazu, resize, anotace (povinné) |
| `numpy` | Maticové operace (povinné) |
| `easyocr` | Deep-learning OCR engine (výchozí); přináší PyTorch |
| `paddleocr` | Alternativní OCR engine |
| `paddlepaddle` | Backend pro PaddleOCR |
| `piexif` | Zápis EXIF XPKeywords v `app.py` (volitelné) |
| `tkinter` | GUI (`app.py`, `gui.py`); standardní knihovna |
| `pytesseract` | Jen pro legacy `gui.py` |

Balíček `pillow`, který uváděla starší verze dokumentace, kód nikde neimportuje.

---

## Tipy pro lepší výsledky

- Fotky v rozlišení alespoň 1000×700 px; číslo musí být i po zmenšení fotky na 1200 px (1024 px u PaddleOCR) vysoké aspoň 12 px
- Startovní číslo čitelné pouhým okem (bez silného pohybového rozmazání)
- Vizuální validace nerozlišuje polaritu – počítá se kontrast, ať jde o tmavé číslice na světlém podkladu, nebo naopak
- EasyOCR je výchozí engine; PaddleOCR je podle poznámek v kódu na CPU výrazně pomalejší

**Co skript záměrně ignoruje:**
- Čísla kratší než 2 cifry a delší než 6 cifer
- Text obsahující cokoli jiného než číslice (např. „A12“)
- Opakující se vzory („0000“, „1111“)

---

## Známá omezení

- Repozitář nemá žádné automatické testy ani CI; jediným testovacím vstupem je `test_race.jpg`.
- Chybí `requirements.txt` a soubor s licencí.
- `--debug` / parametr `debug` nic nedělá.
- `--out` nevytváří složku a při shodě se vstupní složkou přepíše originály (viz výše).
- Zápis metadat v `app.py` ruší dosavadní IPTC data fotky a nerozděluje čísla na samostatná klíčová slova (viz výše).
- Chybějící OCR engine ukončuje proces přes `sys.exit(1)` i při použití `bibnumber.py` jako knihovny.

---

## Licence

Původní projekt [gheinrich/bibnumber](https://github.com/gheinrich/bibnumber) nemá explicitní licenci. Tato Python reimplementace vznikla jako nezávislé dílo pro osobní/výzkumné účely.
