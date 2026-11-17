import os
import queue
import threading
import traceback
import webbrowser

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .config import Settings, load_settings, save_settings
from .corpus import TITLES, build_chunks
from .embed import build_embeddings
from .index import ensure_fts, load_chunks
from .rag import RagEngine

ABOUT = (
    "mia-rag\n\n"
    "Retrieval-augmented generation over 20 classic works of Marxist\n"
    "political economy (Marxists Internet Archive).\n\n"
    "Embeddings: Ollama (nomic-embed-text)\n"
    "Generation: OpenCode Go (deepseek-v4.1-flash)"
)


class SettingsDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app.root)
        self.app = app
        self.title("Settings")
        self.transient(app.root)
        self.resizable(True, False)
        s = app.settings
        self.vars = {
            "corpus_dir": tk.StringVar(value=s.corpus_dir),
            "data_dir": tk.StringVar(value=s.data_dir),
            "api_base": tk.StringVar(value=s.api_base),
            "model": tk.StringVar(value=s.model),
            "api_key": tk.StringVar(value=s.api_key),
            "ollama_url": tk.StringVar(value=s.ollama_url),
            "embed_model": tk.StringVar(value=s.embed_model),
            "top_k": tk.StringVar(value=str(s.top_k)),
        }
        rows = [
            ("Corpus folder", "corpus_dir", True),
            ("Index folder", "data_dir", True),
            ("API base URL", "api_base", False),
            ("Chat model", "model", False),
            ("API key", "api_key", False),
            ("Ollama URL", "ollama_url", False),
            ("Embedding model", "embed_model", False),
            ("Passages (top k)", "top_k", False),
        ]
        for i, (label, key, browse) in enumerate(rows):
            ttk.Label(self, text=label).grid(row=i, column=0, sticky="w", padx=8, pady=4)
            ent = ttk.Entry(self, textvariable=self.vars[key], width=48)
            if key == "api_key":
                ent.configure(show="*")
            ent.grid(row=i, column=1, sticky="ew", padx=4, pady=4)
            if browse:
                ttk.Button(
                    self, text="...", width=3, command=lambda k=key: self.browse(k)
                ).grid(row=i, column=2, padx=4)
        ttk.Label(
            self,
            text="The API key is stored locally with 0600 permissions, never in the repository.",
            foreground="#666",
        ).grid(row=len(rows), column=0, columnspan=3, sticky="w", padx=8)
        bar = ttk.Frame(self)
        bar.grid(row=len(rows) + 1, column=0, columnspan=3, sticky="e", pady=8, padx=8)
        ttk.Button(bar, text="Cancel", command=self.destroy).pack(side="left", padx=4)
        ttk.Button(bar, text="Save", command=self.save).pack(side="left")
        self.columnconfigure(1, weight=1)
        self.grab_set()

    def browse(self, key):
        initial = self.vars[key].get() or os.getcwd()
        path = filedialog.askdirectory(parent=self, initialdir=initial)
        if path:
            self.vars[key].set(path)

    def save(self):
        s = self.app.settings
        s.corpus_dir = self.vars["corpus_dir"].get()
        s.data_dir = self.vars["data_dir"].get()
        s.api_base = self.vars["api_base"].get()
        s.model = self.vars["model"].get()
        s.api_key = self.vars["api_key"].get()
        s.ollama_url = self.vars["ollama_url"].get()
        s.embed_model = self.vars["embed_model"].get()
        try:
            s.top_k = max(1, int(self.vars["top_k"].get()))
        except ValueError:
            messagebox.showerror("Settings", "top k must be an integer", parent=self)
            return
        save_settings(s)
        self.app.engine = None
        self.app.refresh_status()
        self.destroy()


class App:
    def __init__(self, root):
        self.root = root
        self.settings = load_settings()
        self.engine = None
        self.busy = False
        self.queue = queue.Queue()

        root.title("mia-rag — Marxist Economics Reader")
        root.geometry("1180x760")
        root.minsize(900, 600)

        self._build_menu()
        self._build_toolbar()
        self._build_body()
        self._build_status()

        root.after(100, self._poll)
        self.refresh_status()

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Settings...", command=lambda: SettingsDialog(self))
        file_menu.add_command(label="Rebuild index...", command=self.rebuild_index)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.root.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(
            label="About", command=lambda: messagebox.showinfo("About", ABOUT)
        )
        menubar.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menubar)

    def _build_toolbar(self):
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(fill="x")
        self.ask_btn = ttk.Button(bar, text="Ask", command=self.ask)
        self.ask_btn.pack(side="left")
        ttk.Button(bar, text="Settings", command=lambda: SettingsDialog(self)).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(bar, text="Rebuild index", command=self.rebuild_index).pack(
            side="left", padx=(6, 0)
        )
        self.model_label = ttk.Label(bar, text="", foreground="#666")
        self.model_label.pack(side="right")

    def _build_body(self):
        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(paned)
        paned.add(left, weight=1)
        ttk.Label(left, text="Question").pack(anchor="w")
        self.question = ScrolledText(left, height=14, wrap="word", font=("TkDefaultFont", 11))
        self.question.pack(fill="both", expand=False, pady=(2, 8))
        self.question.bind("<Control-Return>", lambda e: self.ask())

        books = ttk.LabelFrame(left, text="Books (none selected = all)")
        books.pack(fill="both", expand=True)
        box = ttk.Frame(books)
        box.pack(fill="both", expand=True, padx=4, pady=4)
        self.book_list = tk.Listbox(
            box, selectmode="extended", exportselection=False, height=12
        )
        sb = ttk.Scrollbar(box, orient="vertical", command=self.book_list.yview)
        self.book_list.configure(yscrollcommand=sb.set)
        self.book_list.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        for title in TITLES.values():
            self.book_list.insert("end", title)
        btns = ttk.Frame(books)
        btns.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Button(
            btns,
            text="All",
            command=lambda: self.book_list.select_set(0, "end"),
        ).pack(side="left")
        ttk.Button(btns, text="None", command=lambda: self.book_list.select_clear(0, "end")).pack(
            side="left", padx=4
        )

        right = ttk.Notebook(paned)
        paned.add(right, weight=3)
        answer_frame = ttk.Frame(right)
        right.add(answer_frame, text="Answer")
        self.answer = ScrolledText(
            answer_frame, wrap="word", font=("TkDefaultFont", 11), state="disabled"
        )
        self.answer.pack(fill="both", expand=True)

        sources_frame = ttk.Frame(right)
        right.add(sources_frame, text="Sources")
        self.sources = ttk.Treeview(
            sources_frame, columns=("n", "title", "url"), show="headings"
        )
        self.sources.heading("n", text="#")
        self.sources.heading("title", text="Book")
        self.sources.heading("url", text="Source URL")
        self.sources.column("n", width=40, anchor="center", stretch=False)
        self.sources.column("title", width=320)
        self.sources.column("url", width=520)
        ssb = ttk.Scrollbar(sources_frame, orient="vertical", command=self.sources.yview)
        self.sources.configure(yscrollcommand=ssb.set)
        self.sources.pack(side="left", fill="both", expand=True)
        ssb.pack(side="right", fill="y")
        self.sources.bind("<Double-1>", self._open_source)

    def _build_status(self):
        bar = ttk.Frame(self.root, padding=(8, 2))
        bar.pack(fill="x", side="bottom")
        self.status = ttk.Label(bar, text="", anchor="w")
        self.status.pack(side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=160)
        self.progress.pack(side="right")

    def _set_status(self, text, busy=None):
        self.status.configure(text=text)
        if busy is None:
            return
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()

        self.busy = busy
        state = "disabled" if busy else "normal"
        self.ask_btn.configure(state=state)

    def refresh_status(self):
        s = self.settings
        self.model_label.configure(text=f"{s.model}  ·  top {s.top_k}")
        missing = [
            name
            for name, path in (
                ("chunks", s.chunks_path),
                ("embeddings", s.embeddings_path),
            )
            if not os.path.exists(path)
        ]
        if missing:
            self._set_status(
                "Index missing ("
                + ", ".join(missing)
                + "). Use Rebuild index or set the corpus folder in Settings."
            )
        else:
            self._set_status(f"Index: {s.data_dir}")

    def selected_titles(self):
        idx = self.book_list.curselection()
        if not idx:
            return None
        return [self.book_list.get(i) for i in idx]

    def ask(self):
        if self.busy:
            return
        q = self.question.get("1.0", "end").strip()
        if not q:
            messagebox.showinfo("mia-rag", "Enter a question first.")
            return
        if not os.path.exists(self.settings.chunks_path):
            messagebox.showerror(
                "mia-rag",
                "No index found. Open Settings and set the corpus folder, then rebuild.",
            )
            return
        self._set_status("Starting...", True)
        self.answer.configure(state="normal")
        self.answer.delete("1.0", "end")
        self.answer.configure(state="disabled")
        for row in self.sources.get_children():
            self.sources.delete(row)

        def work():
            try:
                if self.engine is None:
                    self.engine = RagEngine(self.settings)
                result = self.engine.ask(
                    q,
                    titles=self.selected_titles(),
                    progress=lambda m: self.queue.put(("status", m)),
                )
                self.queue.put(("answer", result))
            except Exception as e:
                self.queue.put(("error", f"{e}\n\n{traceback.format_exc()}"))

        threading.Thread(target=work, daemon=True).start()

    def rebuild_index(self):
        if self.busy:
            return
        s = self.settings
        if not os.path.isdir(s.corpus_dir):
            chosen = filedialog.askdirectory(
                title="Select the folder containing the 20 .txt files"
            )
            if not chosen:
                return
            s.corpus_dir = chosen
        if not messagebox.askyesno(
            "Rebuild index",
            f"Build the index from:\n{s.corpus_dir}\n\ninto:\n{s.data_dir}\n\n"
            "Embedding all chunks with Ollama can take 30-60 minutes on CPU.",
        ):
            return
        self._set_status("Building index...", True)

        def progress(msg):
            self.queue.put(("status", msg))

        def work():
            try:
                os.makedirs(s.data_dir, exist_ok=True)
                progress("Chunking corpus...")
                build_chunks(
                    s.corpus_dir,
                    s.chunks_path,
                    s.chunk_size,
                    s.overlap,
                    progress,
                )
                progress("Embedding chunks...")
                build_embeddings(
                    s.chunks_path,
                    s.embeddings_path,
                    s.embed_model,
                    s.ollama_url,
                    progress=progress,
                )
                progress("Building full-text index...")
                ensure_fts(load_chunks(s.chunks_path), s.fts_path)
                save_settings(s)
                self.engine = None
                self.queue.put(("built", None))
            except Exception as e:
                self.queue.put(("error", f"{e}\n\n{traceback.format_exc()}"))

        threading.Thread(target=work, daemon=True).start()

    def _open_source(self, event):
        sel = self.sources.selection()
        if not sel:
            return
        url = self.sources.item(sel[0], "values")[2]
        if url:
            webbrowser.open(url)

    def _poll(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "status":
                    self._set_status(payload, True)
                elif kind == "answer":
                    self._show_answer(payload)
                elif kind == "built":
                    self._set_status("Index built.", False)
                    self.refresh_status()
                elif kind == "error":
                    self._set_status("Error.", False)
                    messagebox.showerror("mia-rag", payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _show_answer(self, result):
        self.answer.configure(state="normal")
        self.answer.delete("1.0", "end")
        self.answer.insert("1.0", result["answer"])
        self.answer.configure(state="disabled")
        for s in result["sources"]:
            self.sources.insert("", "end", values=(s["n"], s["title"], s["source"]))
        self._set_status(
            "Done. Queries: " + " | ".join(result.get("queries", [])), False
        )


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
