#!/usr/bin/env python3
"""
app.py – GUI pro detekci startovních čísel ze závodních fotek

Spuštění:
    python3 app.py

Funkce:
    • Výběr složky s fotkami
    • Detekce startovních čísel (bibnumber.py)
    • Zápis čísel jako klíčová slova do IPTC metadat fotek (dataset 2:25)
    • Průběžný výpis stavu na příkazový řádek i do okna aplikace

Závislosti:
    pip install easyocr paddleocr piexif pillow opencv-python
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import struct
import sys
from pathlib import Path

try:
    import piexif
    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False

from bibnumber import detect_bibs, is_easyocr_available, is_paddleocr_available

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
JPEG_EXTENSIONS  = {".jpg", ".jpeg"}


# ---------------------------------------------------------------------------
# Zápis IPTC klíčových slov do JPEG
# ---------------------------------------------------------------------------

def _iptc_records(keywords: list) -> bytes:
    """Sestaví jeden IPTC záznam pro klíčová slova (dataset 2:25), čárkou oddělená."""
    data = ", ".join(str(kw) for kw in keywords).encode("utf-8")[:255]
    return b"\x1c\x02\x19" + struct.pack(">H", len(data)) + data


def _app13_segment(iptc_data: bytes) -> bytes:
    """Zabalí IPTC data do APP13 segmentu (Photoshop 3.0 / 8BIM)."""
    # Photoshop 3.0 header + 8BIM blok pro IPTC-NAA (typ 0x0404)
    header = b"Photoshop 3.0\x00" + b"8BIM\x04\x04\x00\x00"
    # IPTC data + padding na sudou délku
    padded = iptc_data + (b"\x00" if len(iptc_data) % 2 else b"")
    content = header + struct.pack(">I", len(iptc_data)) + padded
    return b"\xff\xed" + struct.pack(">H", len(content) + 2) + content


def write_keywords_to_jpeg(path: Path, keywords: list) -> bool:
    """
    Zapíše klíčová slova do JPEG:
      • IPTC dataset 2:25  – čitelné v Lightroomu, Capture One, exiftool, …
      • XPKeywords (EXIF)  – čitelné v Průzkumníku Windows
    Každé startovní číslo je samostatné klíčové slovo.
    Vrátí True při úspěchu.
    """
    if path.suffix.lower() not in JPEG_EXTENSIONS:
        return False
    try:
        raw = path.read_bytes()
        if raw[:2] != b"\xff\xd8":
            return False

        new_app13 = _app13_segment(_iptc_records(keywords))

        # Projdi JPEG segmenty – odstraň staré APP13, ostatní ponech
        out = bytearray(b"\xff\xd8")
        pos = 2
        while pos + 1 < len(raw):
            if raw[pos] != 0xFF:
                out.extend(raw[pos:])
                break
            marker = raw[pos:pos + 2]
            # Standalone markery (nemají délkové pole)
            if marker[1] in (0xD8, 0xD9, 0xD0, 0xD1, 0xD2, 0xD3,
                             0xD4, 0xD5, 0xD6, 0xD7):
                out.extend(marker)
                pos += 2
                continue
            if pos + 4 > len(raw):
                out.extend(raw[pos:])
                break
            seg_len = struct.unpack(">H", raw[pos + 2:pos + 4])[0]
            if marker != b"\xff\xed":          # APP13 vynech, vše ostatní zachovej
                out.extend(raw[pos:pos + 2 + seg_len])
            pos += 2 + seg_len

        # Vlož nový APP13 hned za SOI (první 2 bajty)
        result = bytes(out[:2]) + new_app13 + bytes(out[2:])
        path.write_bytes(result)

        # XPKeywords pro Windows Průzkumník (středník jako oddělovač, UTF-16-LE)
        if PIEXIF_AVAILABLE:
            try:
                exif_dict = piexif.load(str(path))
            except Exception:
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            xp = ", ".join(str(n) for n in keywords) + "\x00"
            exif_dict["0th"][piexif.ImageIFD.XPKeywords] = xp.encode("utf-16-le")
            piexif.insert(piexif.dump(exif_dict), str(path))

        return True

    except Exception as e:
        print(f"  [VAROVÁNÍ] Zápis klíčových slov selhal ({path.name}): {e}", flush=True)
        return False


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Detekce startovních čísel")
        self.root.geometry("760x540")
        self.root.minsize(540, 400)
        self._stop_event = threading.Event()
        self._build_ui()
        self._check_deps()

    # ------------------------------------------------------------------ stavba UI

    def _build_ui(self):
        # --- Horní panel: složka + tlačítka ---
        top = ttk.Frame(self.root, padding=(10, 10, 10, 6))
        top.pack(fill=tk.X)

        ttk.Label(top, text="Složka:").pack(side=tk.LEFT)

        self.folder_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.folder_var, state="readonly",
                  width=50).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))

        ttk.Button(top, text="Procházet…", command=self._pick_folder).pack(side=tk.LEFT)

        self.start_btn = ttk.Button(top, text="Spustit", command=self._start,
                                    state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=(6, 0))

        self.stop_btn = ttk.Button(top, text="Zastavit", command=self._stop,
                                   state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(4, 0))

        # --- Výběr OCR enginu ---
        eng_frame = ttk.Frame(self.root, padding=(10, 0, 10, 6))
        eng_frame.pack(fill=tk.X)

        ttk.Label(eng_frame, text="Engine:").pack(side=tk.LEFT)

        self.engine_var = tk.StringVar(value="easyocr")
        self.engine_var.trace_add("write", self._on_engine_change)
        self.easy_radio = ttk.Radiobutton(
            eng_frame, text="EasyOCR", variable=self.engine_var, value="easyocr"
        )
        self.easy_radio.pack(side=tk.LEFT, padx=(8, 0))
        from bibnumber import _detect_gpu_paddle
        paddle_note = "" if _detect_gpu_paddle() else "  (pomalejší bez GPU)"
        self.paddle_radio = ttk.Radiobutton(
            eng_frame, text=f"PaddleOCR{paddle_note}",
            variable=self.engine_var, value="paddleocr"
        )
        self.paddle_radio.pack(side=tk.LEFT, padx=(8, 0))

        # --- Progress bar ---
        self.progress = ttk.Progressbar(self.root, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, padx=10, pady=(0, 4))

        # --- Log area (terminálový styl) ---
        log_frame = ttk.Frame(self.root)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 4))

        self.log = tk.Text(
            log_frame, wrap=tk.WORD, state=tk.DISABLED,
            bg="#1e1e1e", fg="#d4d4d4",
            font=("Courier", 10), relief=tk.FLAT,
            selectbackground="#264f78"
        )
        sb = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # Barevné tagy pro různé typy zpráv
        self.log.tag_configure("head",  foreground="#dcdcaa", font=("Courier", 10, "bold"))
        self.log.tag_configure("ok",    foreground="#4ec9b0")
        self.log.tag_configure("warn",  foreground="#ce9178")
        self.log.tag_configure("error", foreground="#f44747")
        self.log.tag_configure("info",  foreground="#9cdcfe")
        self.log.tag_configure("dim",   foreground="#808080")

        # --- Stavový řádek ---
        self.status_var = tk.StringVar(value="Vyberte složku a klikněte na Spustit.")
        ttk.Label(self.root, textvariable=self.status_var, anchor=tk.W,
                  padding=(10, 2, 10, 6)).pack(fill=tk.X, side=tk.BOTTOM)

    # ------------------------------------------------------------------ kontroly

    def _check_deps(self):
        easy_ok   = is_easyocr_available()
        paddle_ok = is_paddleocr_available()

        if not easy_ok:
            self.easy_radio.configure(state=tk.DISABLED)
        if not paddle_ok:
            self.paddle_radio.configure(state=tk.DISABLED)

        # Nastavíme výchozí engine na první dostupný
        if not easy_ok and paddle_ok:
            self.engine_var.set("paddleocr")
        elif not easy_ok and not paddle_ok:
            messagebox.showerror(
                "Žádný OCR engine nenalezen",
                "Není nainstalován ani EasyOCR, ani PaddleOCR.\n\n"
                "Nainstalujte alespoň jeden:\n"
                "  pip install easyocr\n"
                "  pip install paddleocr"
            )

    # ------------------------------------------------------------------ log helpers

    def _log(self, text: str, tag: str = ""):
        """Vytiskne zprávu na stdout a zároveň ji přidá do GUI logu."""
        print(text, flush=True)
        self.root.after(0, self._log_append, text, tag)

    def _log_append(self, text: str, tag: str):
        self.log.configure(state=tk.NORMAL)
        if tag:
            self.log.insert(tk.END, text + "\n", tag)
        else:
            self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _log_clear(self):
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)

    def _on_engine_change(self, *_):
        from bibnumber import _detect_gpu_paddle, _detect_gpu_easyocr
        if self.engine_var.get() == "paddleocr":
            if _detect_gpu_paddle():
                self.status_var.set("PaddleOCR: GPU detekováno – rychlost srovnatelná s EasyOCR.")
            else:
                self.status_var.set(
                    "PaddleOCR: GPU nedostupné – CPU je výrazně pomalejší (desítky sekund/fotka)."
                )
        else:
            gpu = _detect_gpu_easyocr()
            self.status_var.set(
                f"EasyOCR {'[GPU]' if gpu else '[CPU]'} – vyberte složku a klikněte na Spustit."
            )

    def _set_status(self, text: str):
        self.root.after(0, self.status_var.set, text)

    def _set_progress(self, value: float):
        self.root.after(0, self.progress.configure, {"value": value})

    # ------------------------------------------------------------------ akce tlačítek

    def _pick_folder(self):
        folder = filedialog.askdirectory(title="Vyberte složku s fotkami")
        if folder:
            self.folder_var.set(folder)
            self.start_btn.configure(state=tk.NORMAL)

    def _start(self):
        folder = self.folder_var.get().strip()
        if not folder:
            return
        self._stop_event.clear()
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self._set_progress(0)
        self._log_clear()
        threading.Thread(target=self._run, args=(folder,), daemon=True).start()

    def _stop(self):
        self._stop_event.set()
        self._log("  [STOP] Čekám na dokončení aktuální fotky…", "warn")
        self.stop_btn.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------ zpracování

    def _run(self, folder: str):
        try:
            self._process(folder)
        except Exception as e:
            self._log(f"\n[FATÁLNÍ CHYBA] {e}", "error")
            self._set_status(f"Chyba: {e}")
        finally:
            self.root.after(0, self.start_btn.configure, {"state": tk.NORMAL})
            self.root.after(0, self.stop_btn.configure,  {"state": tk.DISABLED})

    def _process(self, folder: str):
        engine = self.engine_var.get()

        photos = sorted(
            p for p in Path(folder).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not photos:
            self._log("Ve složce nebyly nalezeny žádné fotky.", "warn")
            self._set_status("Žádné fotky nenalezeny.")
            return

        total = len(photos)

        # Výstupní složka pro anotované kopie
        annotated_dir = Path(folder) / "_annotated"
        annotated_dir.mkdir(exist_ok=True)

        engine_label = "EasyOCR" if engine == "easyocr" else "PaddleOCR"
        self._log(f"{'='*60}", "head")
        self._log(f"  Složka:   {folder}", "head")
        self._log(f"  Fotek:    {total}", "head")
        self._log(f"  Engine:   {engine_label}", "head")
        self._log(f"  Výstup:   {annotated_dir}", "head")
        self._log("  IPTC:     klíčová slova (dataset 2:25)", "head")
        if PIEXIF_AVAILABLE:
            self._log("  EXIF:     XPKeywords (Windows Průzkumník)", "head")
        else:
            self._log("  EXIF XPKeywords: vypnuto – nainstalujte: pip install piexif", "warn")
        self._log(f"{'='*60}", "head")

        ok_count       = 0   # fotek s nalezeným číslem
        kw_count       = 0   # fotek s úspěšným zápisem klíčových slov
        annotated_count = 0  # uložených anotovaných kopií
        total_nums     = 0   # celkem nalezených čísel

        for i, photo in enumerate(photos, 1):
            if self._stop_event.is_set():
                self._log("\n--- Zpracování zastaveno ---", "warn")
                self._set_status("Zastaveno.")
                return

            self._set_status(f"Zpracovávám {i}/{total}: {photo.name}")
            self._log(f"\n[{i}/{total}]  {photo.name}", "info")

            # Detekce čísel + uložení anotované kopie
            try:
                numbers = detect_bibs(str(photo), out_dir=str(annotated_dir),
                                      engine=engine)
            except Exception as e:
                self._log(f"  CHYBA při detekci: {e}", "error")
                self._set_progress(i / total * 100)
                continue

            if numbers:
                nums_str = ", ".join(str(n) for n in numbers)
                self._log(f"  Nalezena čísla:  {nums_str}", "ok")
                ok_count  += 1
                total_nums += len(numbers)

                # Anotovaná kopie
                ann_path = annotated_dir / photo.name
                if ann_path.exists():
                    self._log(f"  Anotovaná kopie → _annotated/{photo.name}", "ok")
                    annotated_count += 1

                # Zápis klíčových slov do metadat
                if photo.suffix.lower() in JPEG_EXTENSIONS:
                    written = write_keywords_to_jpeg(photo, numbers)
                    if written:
                        kw_label = "  |  ".join(str(n) for n in numbers)
                        self._log(f"  Klíčová slova   → {kw_label}", "ok")
                        kw_count += 1
                    else:
                        self._log("  Zápis klíčových slov selhal.", "warn")
                else:
                    self._log(f"  Metadata přeskočena (formát {photo.suffix} nepodporuje IPTC).", "dim")
            else:
                self._log("  Žádná startovní čísla nebyla nalezena.", "dim")

            self._set_progress(i / total * 100)

        # Souhrn
        self._log(f"\n{'='*60}", "head")
        self._log(f"  Zpracováno fotek:         {total}", "head")
        self._log(f"  Fotek s nalezenými čísly: {ok_count}", "head")
        self._log(f"  Celkem nalezených čísel:  {total_nums}", "head")
        self._log(f"  Anotované kopie uloženy:  {annotated_count}  → _annotated/", "head")
        self._log(f"  Klíčová slova zapsána:    {kw_count}", "head")
        self._log(f"{'='*60}", "head")

        self._set_status(
            f"Hotovo – {ok_count}/{total} fotek s čísly, "
            f"{annotated_count} anotovaných kopií v _annotated/."
        )


# ---------------------------------------------------------------------------
# Vstupní bod
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
