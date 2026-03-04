#!/usr/bin/env python3
"""
GUI pro detekci startovních čísel – vyberte složku s fotkami
a program zobrazí nalezená čísla u každé fotky.
"""

import tkinter as tk
from tkinter import ttk, filedialog
import threading
from pathlib import Path

from bibnumber import detect_bibs

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class BibGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Detekce startovních čísel")
        self.root.geometry("700x500")
        self.root.minsize(500, 350)

        self._build_ui()

    def _build_ui(self):
        # --- Horní panel: výběr složky ---
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        self.folder_var = tk.StringVar(value="(žádná složka)")
        ttk.Label(top, textvariable=self.folder_var, anchor=tk.W).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        ttk.Button(top, text="Vybrat složku…", command=self._pick_folder).pack(
            side=tk.RIGHT, padx=(10, 0)
        )

        # --- Tabulka s výsledky ---
        cols = ("file", "numbers")
        self.tree = ttk.Treeview(self.root, columns=cols, show="headings")
        self.tree.heading("file", text="Soubor")
        self.tree.heading("numbers", text="Startovní čísla")
        self.tree.column("file", width=280)
        self.tree.column("numbers", width=380)

        scrollbar = ttk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=(0, 10))
        scrollbar.pack(side=tk.LEFT, fill=tk.Y, pady=(0, 10), padx=(0, 10))

        # --- Stavový řádek ---
        self.status_var = tk.StringVar(value="Vyberte složku s fotkami.")
        ttk.Label(self.root, textvariable=self.status_var, anchor=tk.W, padding=5).pack(
            side=tk.BOTTOM, fill=tk.X
        )

    def _pick_folder(self):
        folder = filedialog.askdirectory(title="Vyberte složku s fotkami")
        if not folder:
            return

        self.folder_var.set(folder)
        self.tree.delete(*self.tree.get_children())
        self.status_var.set("Zpracovávám…")
        self.root.update_idletasks()

        # Spustí zpracování v jiném vlákně, aby GUI nezamrzlo
        threading.Thread(target=self._process_folder, args=(folder,), daemon=True).start()

    def _process_folder(self, folder: str):
        photos = sorted(
            p for p in Path(folder).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not photos:
            self.root.after(0, self.status_var.set, "Ve složce nebyly nalezeny žádné fotky.")
            return

        for i, photo in enumerate(photos, 1):
            self.root.after(
                0, self.status_var.set, f"Zpracovávám {i}/{len(photos)}: {photo.name}"
            )

            try:
                numbers = detect_bibs(str(photo))
            except Exception as e:
                numbers = []
                print(f"[CHYBA] {photo.name}: {e}")

            display = ", ".join(str(n) for n in numbers) if numbers else "–"
            self.root.after(0, self.tree.insert, "", tk.END, None, {"values": (photo.name, display)})

        self.root.after(
            0, self.status_var.set, f"Hotovo – zpracováno {len(photos)} fotek."
        )


def main():
    root = tk.Tk()
    BibGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
