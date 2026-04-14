#!/usr/bin/env python3
"""
app.py – GUI pro detekci startovních čísel ze závodních fotek

Spuštění:
    python3 app.py

Funkce:
    • Výběr složky s fotkami
    • Detekce startovních čísel (bibnumber.py)
    • Zápis čísel do EXIF metadat fotek (pole ImageDescription)
    • Průběžný výpis stavu na příkazový řádek i do okna aplikace

Závislosti:
    pip install piexif pillow opencv-python pytesseract numpy scipy
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import sys
from pathlib import Path

try:
    import piexif
    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False

from bibnumber import detect_bibs, check_tesseract

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
JPEG_EXTENSIONS  = {".jpg", ".jpeg"}


# ---------------------------------------------------------------------------
# EXIF zápis
# ---------------------------------------------------------------------------

def write_bib_to_exif(image_path: Path, numbers: list) -> bool:
    """
    Zapíše detekovaná startovní čísla do EXIF pole ImageDescription.
    Vrátí True při úspěchu.
    Funguje pouze pro JPEG soubory (omezení piexif).
    """
    if not PIEXIF_AVAILABLE:
        return False
    if image_path.suffix.lower() not in JPEG_EXTENSIONS:
        return False

    numbers_str = ", ".join(str(n) for n in sorted(numbers))
    try:
        try:
            exif_dict = piexif.load(str(image_path))
        except Exception:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

        exif_dict["0th"][piexif.ImageIFD.ImageDescription] = numbers_str.encode("utf-8")

        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, str(image_path))
        return True
    except Exception as e:
        print(f"  [VAROVÁNÍ] EXIF zápis selhal ({image_path.name}): {e}", flush=True)
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
        self._check_tesseract()

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

    def _check_tesseract(self):
        try:
            check_tesseract()
        except SystemExit:
            messagebox.showerror(
                "Tesseract nenalezen",
                "Tesseract OCR není nainstalován nebo není v PATH.\n\n"
                "Windows:  https://github.com/UB-Mannheim/tesseract/wiki\n"
                "macOS:    brew install tesseract\n"
                "Linux:    sudo apt install tesseract-ocr\n\n"
                "Bez Tesseractu nelze rozpoznávat čísla."
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
        photos = sorted(
            p for p in Path(folder).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not photos:
            self._log("Ve složce nebyly nalezeny žádné fotky.", "warn")
            self._set_status("Žádné fotky nenalezeny.")
            return

        total = len(photos)
        self._log(f"{'='*60}", "head")
        self._log(f"  Složka:  {folder}", "head")
        self._log(f"  Fotek:   {total}", "head")
        if PIEXIF_AVAILABLE:
            self._log("  EXIF:    zapnuto (piexif)", "head")
        else:
            self._log("  EXIF:    vypnuto – nainstalujte: pip install piexif", "warn")
        self._log(f"{'='*60}", "head")

        ok_count    = 0   # fotek s nalezeným číslem
        exif_count  = 0   # fotek s úspěšným EXIF zápisem
        total_nums  = 0   # celkem nalezených čísel

        for i, photo in enumerate(photos, 1):
            if self._stop_event.is_set():
                self._log("\n--- Zpracování zastaveno ---", "warn")
                self._set_status("Zastaveno.")
                return

            self._set_status(f"Zpracovávám {i}/{total}: {photo.name}")
            self._log(f"\n[{i}/{total}]  {photo.name}", "info")

            # Detekce čísel
            try:
                numbers = detect_bibs(str(photo))
            except Exception as e:
                self._log(f"  CHYBA při detekci: {e}", "error")
                self._set_progress(i / total * 100)
                continue

            if numbers:
                nums_str = ", ".join(str(n) for n in numbers)
                self._log(f"  Nalezena čísla:  {nums_str}", "ok")
                ok_count  += 1
                total_nums += len(numbers)

                # Zápis do EXIF metadat
                if photo.suffix.lower() in JPEG_EXTENSIONS:
                    if PIEXIF_AVAILABLE:
                        written = write_bib_to_exif(photo, numbers)
                        if written:
                            self._log(f"  EXIF zapsán  →  ImageDescription = \"{nums_str}\"", "ok")
                            exif_count += 1
                        else:
                            self._log("  EXIF zápis selhal.", "warn")
                    else:
                        self._log("  EXIF přeskočeno (piexif není nainstalován).", "dim")
                else:
                    self._log(f"  EXIF přeskočeno (formát {photo.suffix} nepodporuje EXIF).", "dim")
            else:
                self._log("  Žádná startovní čísla nebyla nalezena.", "dim")

            self._set_progress(i / total * 100)

        # Souhrn
        self._log(f"\n{'='*60}", "head")
        self._log(f"  Zpracováno fotek:        {total}", "head")
        self._log(f"  Fotek s čísly:           {ok_count}", "head")
        self._log(f"  Celkem nalezených čísel: {total_nums}", "head")
        if PIEXIF_AVAILABLE:
            self._log(f"  EXIF zapsán do fotek:    {exif_count}", "head")
        self._log(f"{'='*60}", "head")

        self._set_status(
            f"Hotovo – {ok_count}/{total} fotek s čísly, "
            f"celkem {total_nums} čísel"
            + (f", EXIF zapsán do {exif_count} fotek" if PIEXIF_AVAILABLE else "")
            + "."
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
