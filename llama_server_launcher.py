from __future__ import annotations

import ctypes
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from tkinter import filedialog
from typing import Any

import customtkinter as ctk


APP_NAME = "Llama Server Launcher"
ACCENT = "#39C6B0"
BACKGROUND = "#101214"
SURFACE = "#191D20"
SURFACE_ALT = "#202529"
TEXT = "#E7ECEF"
MUTED = "#98A3AA"
GREEN = "#40C98A"
RED = "#F06B72"
YELLOW = "#E9B85C"

KV_CACHE_OPTIONS = [
    "f32",
    "f16",
    "bf16",
    "q8_0",
    "q4_0",
    "q4_1",
    "iq4_nl",
    "q5_0",
    "q5_1",
]


def settings_file() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    return base / "LlamaServerLauncher" / "settings.json"


def application_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def windows_job_for_process(process: subprocess.Popen[str]) -> int | None:
    """Create a kill-on-close job so server child processes cannot outlive the app."""
    if os.name != "nt":
        return None

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimitInformation),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    kernel32.SetInformationJobObject.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]

    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        return None

    info = ExtendedLimitInformation()
    info.BasicLimitInformation.LimitFlags = 0x00002000
    configured = kernel32.SetInformationJobObject(
        handle, 9, ctypes.byref(info), ctypes.sizeof(info)
    )
    assigned = configured and kernel32.AssignProcessToJobObject(
        handle, ctypes.c_void_p(process._handle)
    )
    if not assigned:
        kernel32.CloseHandle(handle)
        return None

    return int(handle)


def close_windows_job(handle: int | None) -> None:
    if handle and os.name == "nt":
        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(
            ctypes.c_void_p(handle)
        )


class LlamaServerLauncher(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title(APP_NAME)
        self.geometry("1200x820")
        self.minsize(980, 700)
        self.configure(fg_color=BACKGROUND)

        self._settings_path = settings_file()
        self._settings = self._load_settings()
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._process: subprocess.Popen[str] | None = None
        self._job_handle: int | None = None
        self._stop_in_progress = False
        self._close_requested = False
        self._restart_after_stop = False
        self._model_choices: dict[str, Path] = {}
        self._model_dirs: list[str] = [
            item
            for item in self._settings.get("model_dirs", [])
            if isinstance(item, str)
        ]
        legacy_model = self._settings.get("model", "")
        if isinstance(legacy_model, str) and legacy_model:
            legacy_path = Path(legacy_model).expanduser()
            legacy_dir = legacy_path if legacy_path.is_dir() else legacy_path.parent
            if str(legacy_dir) not in self._model_dirs:
                self._model_dirs.append(str(legacy_dir))
        self._profiles: dict[str, dict[str, str]] = {
            str(key): value
            for key, value in self._settings.get("profiles", {}).items()
            if isinstance(value, dict)
        }
        self._active_model_path = ""
        self._migrate_legacy_profile()

        self.model_var = ctk.StringVar(value=self._settings.get("model", ""))
        self.server_var = ctk.StringVar(value=self._settings.get("server", ""))
        self.context_var = ctk.StringVar(
            value=str(self._settings.get("context", "32768"))
        )
        self.gpu_layers_var = ctk.StringVar(
            value=str(self._settings.get("gpu_layers", "99"))
        )
        self.cache_k_var = ctk.StringVar(
            value=self._settings.get("cache_k", "q8_0")
        )
        self.cache_v_var = ctk.StringVar(
            value=self._settings.get("cache_v", "q8_0")
        )
        self.port_var = ctk.StringVar(value=str(self._settings.get("port", "8080")))
        self.model_choice_var = ctk.StringVar(value="")

        self._build_ui()
        self._refresh_model_choices()
        self._select_initial_model()
        self.model_entry.bind("<FocusOut>", self._model_entry_focus_out)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._drain_events)
        self.after(200, self._poll_process)

        self._append_log(
            "Готово. Выберите установленную GGUF-модель в списке или добавьте папку с моделями."
        )

    def _load_settings(self) -> dict[str, Any]:
        try:
            return json.loads(self._settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_settings(self) -> None:
        self._capture_active_profile()
        data = {
            "model": self.model_var.get().strip(),
            "server": self.server_var.get().strip(),
            "model_dirs": self._model_dirs,
            "profiles": self._profiles,
        }
        try:
            self._settings_path.parent.mkdir(parents=True, exist_ok=True)
            self._settings_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            self._append_log(f"Не удалось сохранить настройки: {exc}")

    def _migrate_legacy_profile(self) -> None:
        model = self._settings.get("model", "")
        if not isinstance(model, str) or not model:
            return
        model_path = Path(model).expanduser()
        if not model_path.is_file() or str(model_path) in self._profiles:
            return
        self._profiles[str(model_path.resolve())] = {
            "context": str(self._settings.get("context", "32768")),
            "gpu_layers": str(self._settings.get("gpu_layers", "99")),
            "cache_k": str(self._settings.get("cache_k", "q8_0")),
            "cache_v": str(self._settings.get("cache_v", "q8_0")),
            "port": str(self._settings.get("port", "8080")),
        }

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(20, 14))
        header.grid_columnconfigure(0, weight=1)

        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            title_box,
            text="Llama Server",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=TEXT,
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_box,
            text="Локальный запуск GGUF-моделей",
            font=ctk.CTkFont(size=13),
            text_color=MUTED,
        ).pack(anchor="w", pady=(2, 0))

        self.status_badge = ctk.CTkFrame(
            header, fg_color=SURFACE_ALT, corner_radius=8
        )
        self.status_badge.grid(row=0, column=1, sticky="e", padx=(12, 0))
        self.status_dot = ctk.CTkLabel(
            self.status_badge,
            text="●",
            text_color=RED,
            font=ctk.CTkFont(size=13),
        )
        self.status_dot.pack(side="left", padx=(12, 6), pady=9)
        self.status_text = ctk.CTkLabel(
            self.status_badge,
            text="Остановлен",
            text_color=TEXT,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.status_text.pack(side="left", padx=(0, 12), pady=9)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 16))
        body.grid_columnconfigure(0, minsize=390, weight=0)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self._build_settings_panel(body)
        self._build_log_panel(body)

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=24, pady=(0, 20))
        footer.grid_columnconfigure(0, weight=1)

        self.endpoint_label = ctk.CTkLabel(
            footer,
            text="Адрес сервера: http://127.0.0.1:8080",
            text_color=MUTED,
            font=ctk.CTkFont(size=12),
        )
        self.endpoint_label.grid(row=0, column=0, sticky="w")
        buttons = ctk.CTkFrame(footer, fg_color="transparent")
        buttons.grid(row=0, column=1, sticky="e")

        self.start_button = ctk.CTkButton(
            buttons,
            text="▶  Запустить сервер",
            height=40,
            width=172,
            corner_radius=8,
            fg_color=ACCENT,
            hover_color="#2AAE9B",
            text_color="#071B18",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._start_server,
        )
        self.start_button.pack(side="left", padx=(0, 8))
        self.restart_button = ctk.CTkButton(
            buttons,
            text="↻  Перезапустить",
            width=132,
            height=40,
            corner_radius=8,
            fg_color=SURFACE_ALT,
            hover_color="#30383D",
            text_color=TEXT,
            font=ctk.CTkFont(size=18),
            command=self._restart_server,
        )
        self.restart_button.pack(side="left", padx=(0, 8))
        self.restart_button.configure(state="disabled")
        self.stop_button = ctk.CTkButton(
            buttons,
            text="■  Остановить",
            width=132,
            height=40,
            corner_radius=8,
            fg_color=SURFACE_ALT,
            hover_color="#30383D",
            text_color=TEXT,
            command=self._stop_server,
        )
        self.stop_button.pack(side="left")
        self.stop_button.configure(state="disabled")

        self.port_var.trace_add("write", self._update_endpoint)

    def _build_settings_panel(self, parent: ctk.CTkFrame) -> None:
        panel = ctk.CTkFrame(parent, fg_color=SURFACE, corner_radius=10)
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            panel,
            text="Настройки запуска",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(18, 8))

        content = ctk.CTkScrollableFrame(
            panel,
            fg_color="transparent",
            scrollbar_button_color="#384247",
            scrollbar_button_hover_color="#4C595F",
        )
        content.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 12))
        content.grid_columnconfigure(0, weight=1)

        self.model_entry = self._labeled_entry(
            content, "Папка с моделями или путь к GGUF", self.model_var, 0
        )
        model_actions = ctk.CTkFrame(content, fg_color="transparent")
        model_actions.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        model_actions.grid_columnconfigure(0, weight=1)
        model_actions.grid_columnconfigure(1, weight=1)
        model_actions.grid_columnconfigure(2, weight=1)
        self._small_button(
            model_actions, "Добавить папку", self._browse_model_folder
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._small_button(
            model_actions, "Выбрать файл", self._browse_model_file
        ).grid(row=0, column=1, sticky="ew", padx=4)
        self._small_button(
            model_actions, "Обновить", self._refresh_model_choices
        ).grid(row=0, column=2, sticky="ew", padx=(4, 0))

        ctk.CTkLabel(
            content,
            text="Установленные модели",
            text_color=MUTED,
            font=ctk.CTkFont(size=12),
        ).grid(row=3, column=0, sticky="w", padx=8, pady=(2, 4))
        self.model_combo = ctk.CTkComboBox(
            content,
            variable=self.model_choice_var,
            values=["Выберите папку с GGUF"],
            state="disabled",
            height=36,
            corner_radius=7,
            fg_color=SURFACE_ALT,
            border_color="#343D42",
            button_color="#343D42",
            button_hover_color="#465258",
            text_color=TEXT,
            font=ctk.CTkFont(size=12),
            command=self._on_model_selected,
        )
        self.model_combo.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 12))

        self._labeled_path(
            content,
            "Исполняемый файл llama-server",
            self.server_var,
            self._browse_server,
            5,
        )

        grid = ctk.CTkFrame(content, fg_color="transparent")
        grid.grid(row=7, column=0, sticky="ew", padx=4, pady=(2, 0))
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(1, weight=1)
        self._labeled_entry(grid, "Контекст (-c)", self.context_var, 0, column=0)
        self._labeled_entry(
            grid, "Слои GPU (-ngl)", self.gpu_layers_var, 0, column=1
        )

        self._labeled_combo(
            content, "KV-кэш ключей (-ctk)", self.cache_k_var, KV_CACHE_OPTIONS, 8
        )
        self._labeled_combo(
            content, "KV-кэш значений (-ctv)", self.cache_v_var, KV_CACHE_OPTIONS, 10
        )

        port_row = ctk.CTkFrame(content, fg_color="transparent")
        port_row.grid(row=12, column=0, sticky="ew", padx=4, pady=(2, 12))
        port_row.grid_columnconfigure(0, weight=1)
        self._labeled_entry(port_row, "Порт", self.port_var, 0, column=0)

        ctk.CTkLabel(
            content,
            text="Параметры сохраняются на этом компьютере.",
            text_color="#77838A",
            font=ctk.CTkFont(size=11),
        ).grid(row=13, column=0, sticky="w", padx=8, pady=(4, 12))

    def _build_log_panel(self, parent: ctk.CTkFrame) -> None:
        panel = ctk.CTkFrame(parent, fg_color=SURFACE, corner_radius=10)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(panel, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 10))
        top.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            top,
            text="Журнал сервера",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, sticky="w")
        self.clear_log_button = ctk.CTkButton(
            top,
            text="Очистить",
            width=76,
            height=28,
            corner_radius=6,
            fg_color=SURFACE_ALT,
            hover_color="#30383D",
            text_color=MUTED,
            font=ctk.CTkFont(size=11),
            command=self._clear_log,
        )
        self.clear_log_button.grid(row=0, column=1, sticky="e")

        self.log_box = ctk.CTkTextbox(
            panel,
            fg_color="#0C0F10",
            text_color="#D1D9DD",
            corner_radius=8,
            border_width=1,
            border_color="#293135",
            font=ctk.CTkFont(family="Cascadia Code", size=12),
            wrap="word",
            activate_scrollbars=True,
        )
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        self.log_box.configure(state="disabled")

    def _labeled_entry(
        self,
        parent: ctk.CTkFrame,
        label: str,
        variable: ctk.StringVar,
        row: int,
        column: int = 0,
    ) -> ctk.CTkEntry:
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.grid(row=row, column=column, sticky="ew", padx=8, pady=(2, 8))
        wrap.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            wrap,
            text=label,
            text_color=MUTED,
            font=ctk.CTkFont(size=12),
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        entry = ctk.CTkEntry(
            wrap,
            textvariable=variable,
            height=36,
            corner_radius=7,
            fg_color=SURFACE_ALT,
            border_color="#343D42",
            text_color=TEXT,
            font=ctk.CTkFont(size=12),
        )
        entry.grid(row=1, column=0, sticky="ew")
        return entry

    def _labeled_path(
        self,
        parent: ctk.CTkFrame,
        label: str,
        variable: ctk.StringVar,
        browse_command: Any,
        row: int,
    ) -> None:
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.grid(row=row, column=0, sticky="ew", padx=8, pady=(2, 8))
        wrap.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            wrap, text=label, text_color=MUTED, font=ctk.CTkFont(size=12)
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        ctk.CTkEntry(
            wrap,
            textvariable=variable,
            height=36,
            corner_radius=7,
            fg_color=SURFACE_ALT,
            border_color="#343D42",
            text_color=TEXT,
            font=ctk.CTkFont(size=12),
        ).grid(row=1, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(
            wrap,
            text="Обзор",
            width=70,
            height=36,
            corner_radius=7,
            fg_color=SURFACE_ALT,
            hover_color="#30383D",
            text_color=TEXT,
            command=browse_command,
        ).grid(row=1, column=1)

    def _labeled_combo(
        self,
        parent: ctk.CTkFrame,
        label: str,
        variable: ctk.StringVar,
        values: list[str],
        row: int,
    ) -> None:
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.grid(row=row, column=0, sticky="ew", padx=8, pady=(2, 8))
        wrap.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            wrap, text=label, text_color=MUTED, font=ctk.CTkFont(size=12)
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        ctk.CTkComboBox(
            wrap,
            variable=variable,
            values=values,
            height=36,
            corner_radius=7,
            fg_color=SURFACE_ALT,
            border_color="#343D42",
            button_color="#343D42",
            button_hover_color="#465258",
            text_color=TEXT,
            font=ctk.CTkFont(size=12),
            state="readonly",
        ).grid(row=1, column=0, sticky="ew")

    @staticmethod
    def _small_button(parent: ctk.CTkFrame, text: str, command: Any) -> ctk.CTkButton:
        return ctk.CTkButton(
            parent,
            text=text,
            height=32,
            corner_radius=7,
            fg_color=SURFACE_ALT,
            hover_color="#30383D",
            text_color=TEXT,
            font=ctk.CTkFont(size=11),
            command=command,
        )

    def _browse_model_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Выберите GGUF-модель",
            filetypes=[("Модели GGUF", "*.gguf"), ("Все файлы", "*.*")],
        )
        if selected:
            self._register_model_directory(Path(selected).parent)
            self._refresh_model_choices(select_path=Path(selected))

    def _browse_model_folder(self) -> None:
        selected = filedialog.askdirectory(title="Выберите папку с GGUF-моделями")
        if selected:
            self._register_model_directory(Path(selected))
            self._refresh_model_choices()

    def _browse_server(self) -> None:
        selected = filedialog.askopenfilename(
            title="Выберите llama-server",
            filetypes=[
                ("llama-server", "llama-server.exe"),
                ("Исполняемые файлы", "*.exe"),
                ("Все файлы", "*.*"),
            ],
        )
        if selected:
            self.server_var.set(selected)

    def _register_model_directory(self, path: Path) -> None:
        path = path.expanduser().resolve()
        if path.is_dir() and str(path) not in self._model_dirs:
            self._model_dirs.append(str(path))
            self._save_settings()

    def _default_model_directories(self) -> list[Path]:
        candidates = [
            application_directory() / "models",
            application_directory(),
            Path.cwd() / "models",
            Path.home() / "Models",
            Path.home() / "models",
        ]
        result: list[Path] = []
        for candidate in candidates:
            candidate = candidate.expanduser()
            if candidate.is_dir() and candidate not in result:
                result.append(candidate)
        return result

    def _all_model_directories(self) -> list[Path]:
        result = self._default_model_directories()
        for raw in self._model_dirs:
            path = Path(raw).expanduser()
            if path.is_dir() and path not in result:
                result.append(path)
        return result

    def _model_label(self, path: Path) -> str:
        try:
            return str(path.relative_to(application_directory()))
        except ValueError:
            return str(path)

    @staticmethod
    def _is_huggingface_cache_path(path: Path) -> bool:
        return any(part.casefold() in {"snapshots", "blobs"} for part in path.parts)

    @staticmethod
    def _is_blob_path(path: Path) -> bool:
        return any(part.casefold() == "blobs" for part in path.parts)

    @staticmethod
    def _path_key(path: Path) -> str:
        # abspath normalizes the path without following Hugging Face snapshot
        # symlinks into the extensionless blobs directory.
        return os.path.normcase(os.path.abspath(path))

    def _snapshot_gguf_files(self, path: Path) -> list[Path]:
        """Return only real GGUF files from a Hugging Face cache structure."""
        path = path.expanduser()
        if not path.exists():
            return []

        snapshot_roots: list[Path] = []
        current = path if path.is_dir() else path.parent
        while current != current.parent:
            if current.name.casefold() == "snapshots":
                snapshot_roots.append(current)
                break
            if (current / "snapshots").is_dir():
                snapshot_roots.append(current / "snapshots")
                break
            current = current.parent

        found: list[Path] = []
        seen: set[str] = set()
        for root in snapshot_roots:
            try:
                candidates = root.rglob("*.gguf")
            except OSError:
                continue
            for item in candidates:
                if not item.is_file() or self._is_blob_path(item):
                    continue
                key = self._path_key(item)
                if key not in seen:
                    seen.add(key)
                    found.append(Path(os.path.abspath(item)))
        return found

    def _gguf_files_in_directory(self, directory: Path) -> list[Path]:
        """Scan a normal model folder, excluding HF blobs and metadata files."""
        directory = directory.expanduser()
        if not directory.is_dir():
            return []
        is_hf_repo_root = directory.name.casefold().startswith("models--")
        if self._is_huggingface_cache_path(directory) or is_hf_repo_root or (
            (directory / "snapshots").is_dir()
            or (directory / "blobs").is_dir()
        ):
            return self._snapshot_gguf_files(directory)

        found: list[Path] = []
        try:
            candidates = directory.rglob("*.gguf")
        except OSError:
            return found
        for item in candidates:
            if item.is_file() and not self._is_blob_path(item):
                found.append(Path(os.path.abspath(item)))
        return found

    def _resolve_cached_model_path(self, path: Path) -> Path | None:
        """Convert an old HF blob path or cache directory to a snapshots GGUF path."""
        path = path.expanduser()
        if path.is_file() and path.suffix.casefold() == ".gguf" and not self._is_blob_path(path):
            return Path(os.path.abspath(path))

        if not path.exists():
            return None

        if path.is_file() and self._is_blob_path(path):
            cache_root = path.parent.parent
            candidates = self._snapshot_gguf_files(cache_root)
            return candidates[0] if candidates else None

        candidates = self._gguf_files_in_directory(path)
        return candidates[0] if candidates else None

    def _refresh_model_choices(self, select_path: Path | None = None) -> None:
        self._model_choices = {}
        models: list[Path] = []
        seen: set[str] = set()

        for directory in self._all_model_directories():
            for item in self._gguf_files_in_directory(directory):
                key = self._path_key(item)
                if key not in seen:
                    seen.add(key)
                    models.append(Path(os.path.abspath(item)))

        raw = self.model_var.get().strip().strip('"')
        explicit_path = Path(raw).expanduser() if raw else None
        if explicit_path:
            explicit_model = self._resolve_cached_model_path(explicit_path)
            if explicit_model:
                if self._path_key(explicit_model) not in seen:
                    models.append(explicit_model)
                if self._path_key(explicit_model) != self._path_key(explicit_path):
                    self.model_var.set(str(explicit_model))
                    select_path = explicit_model

        models.sort(key=lambda item: str(item).casefold())
        if not models:
            self.model_combo.configure(
                values=["GGUF-модели не найдены"], state="disabled"
            )
            self.model_choice_var.set("GGUF-модели не найдены")
            return

        labels: list[str] = []
        for item in models:
            label = self._model_label(item)
            if label in self._model_choices:
                label = str(item)
            labels.append(label)
            self._model_choices[label] = item

        self.model_combo.configure(values=labels, state="readonly")
        target = select_path or (Path(self._active_model_path) if self._active_model_path else explicit_path)
        selected_label = next(
            (
                label
                for label, path in self._model_choices.items()
                if target and self._path_key(path) == self._path_key(target)
            ),
            labels[0],
        )
        self.model_choice_var.set(selected_label)
        self._on_model_selected(selected_label)

    def _select_initial_model(self) -> None:
        requested = self._settings.get("model", "")
        if isinstance(requested, str) and requested:
            requested_path = self._resolve_cached_model_path(Path(requested))
            if requested_path:
                self._refresh_model_choices(select_path=requested_path)
                return
        if self._model_choices:
            self._on_model_selected(self.model_choice_var.get())

    def _model_entry_focus_out(self, _event: Any = None) -> None:
        raw = self.model_var.get().strip().strip('"')
        path = Path(raw).expanduser() if raw else None
        if path and path.is_dir():
            self._register_model_directory(path)
        selected_path = self._resolve_cached_model_path(path) if path else None
        self._refresh_model_choices(select_path=selected_path)

    def _on_model_selected(self, value: str) -> None:
        selected = self._model_choices.get(value)
        if selected is None:
            return
        self._capture_active_profile()
        self._active_model_path = str(selected)
        # Always keep the actual GGUF from snapshots, never a Hugging Face blob.
        self.model_var.set(str(selected))
        self._load_profile(selected)

    def _profile_values(self) -> dict[str, str]:
        return {
            "context": self.context_var.get().strip(),
            "gpu_layers": self.gpu_layers_var.get().strip(),
            "cache_k": self.cache_k_var.get(),
            "cache_v": self.cache_v_var.get(),
            "port": self.port_var.get().strip(),
        }

    def _capture_active_profile(self) -> None:
        if self._active_model_path:
            self._profiles[self._active_model_path] = self._profile_values()

    def _load_profile(self, model: Path) -> None:
        profile = self._profiles.get(str(model), {})
        self.context_var.set(str(profile.get("context", "32768")))
        self.gpu_layers_var.set(str(profile.get("gpu_layers", "99")))
        self.cache_k_var.set(profile.get("cache_k", "q8_0"))
        self.cache_v_var.set(profile.get("cache_v", "q8_0"))
        self.port_var.set(str(profile.get("port", "8080")))

    def _resolve_model(self) -> Path:
        raw = self.model_var.get().strip().strip('"')
        if not raw:
            raise ValueError("Укажите файл модели GGUF или папку с моделями.")
        path = Path(raw).expanduser()
        resolved = self._resolve_cached_model_path(path)
        if resolved:
            if not path.exists() or self._path_key(resolved) != self._path_key(path):
                self.model_var.set(str(resolved))
            return resolved

        if path.exists() and self._is_blob_path(path):
            raise ValueError(
                "Выбран файл из папки blobs. GGUF нужно брать из snapshots."
            )
        if path.is_file():
            raise ValueError("Файл модели должен иметь расширение .gguf.")
        raise ValueError("В выбранной папке не найден файл .gguf в snapshots.")

    def _resolve_server(self) -> Path:
        raw = self.server_var.get().strip().strip('"')
        if raw:
            path = Path(raw).expanduser()
            if path.is_file():
                return path.resolve()
            raise ValueError("Файл llama-server не найден.")
        local_candidates = [
            application_directory() / "llama-server.exe",
            application_directory() / "llama-server",
            application_directory() / "bin" / "llama-server.exe",
        ]
        found = next((item for item in local_candidates if item.is_file()), None)
        found = found or shutil.which("llama-server.exe") or shutil.which("llama-server")
        if found:
            found_path = Path(found).resolve()
            self.server_var.set(str(found_path))
            return found_path
        raise ValueError(
            "Положите llama-server.exe рядом с EXE или укажите его путь через «Обзор»."
        )

    @staticmethod
    def _positive_int(value: str, name: str, allow_minus_one: bool = False) -> int:
        try:
            result = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"Параметр «{name}» должен быть целым числом.") from exc
        minimum = -1 if allow_minus_one else 1
        if result < minimum:
            raise ValueError(f"Параметр «{name}» должен быть не меньше {minimum}.")
        return result

    def _make_command(self) -> list[str]:
        model = self._resolve_model()
        server = self._resolve_server()
        context = self._positive_int(self.context_var.get(), "Контекст")
        gpu_layers = self._positive_int(
            self.gpu_layers_var.get(), "Слои GPU", allow_minus_one=True
        )
        port = self._positive_int(self.port_var.get(), "Порт")
        if port > 65535:
            raise ValueError("Порт должен быть в диапазоне от 1 до 65535.")

        return [
            str(server),
            "-m",
            str(model),
            "-c",
            str(context),
            "-ngl",
            str(gpu_layers),
            "-ctk",
            self.cache_k_var.get(),
            "-ctv",
            self.cache_v_var.get(),
            "--port",
            str(port),
        ]

    def _start_server(self) -> None:
        if self._process is not None or self._stop_in_progress:
            return
        try:
            command = self._make_command()
        except (OSError, ValueError) as exc:
            self._append_log(f"Ошибка настроек: {exc}")
            return

        self._save_settings()
        self._append_log(f"[{self._timestamp()}] Запуск сервера на порту {self.port_var.get()}...")
        options: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
            "shell": False,
        }
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            options["start_new_session"] = True

        try:
            self._process = subprocess.Popen(command, **options)
        except OSError as exc:
            self._process = None
            self._set_status("Ошибка запуска", RED)
            self._append_log(f"Не удалось запустить llama-server: {exc}")
            return

        self._job_handle = windows_job_for_process(self._process)
        if os.name == "nt" and self._job_handle is None:
            self._append_log(
                "Предупреждение: Windows Job Object недоступен; при остановке будет использован taskkill."
            )
        self._set_status("Запускается", YELLOW)
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.restart_button.configure(state="normal")
        reader = threading.Thread(
            target=self._read_output,
            args=(self._process,),
            name="llama-server-log-reader",
            daemon=True,
        )
        reader.start()

    def _read_output(self, process: subprocess.Popen[str]) -> None:
        stream = process.stdout
        if stream is None:
            return
        try:
            for line in stream:
                self._events.put(("log", line.rstrip("\r\n")))
        except (OSError, ValueError) as exc:
            self._events.put(("log", f"Ошибка чтения вывода: {exc}"))
        finally:
            stream.close()

    def _stop_server(self) -> None:
        self._restart_after_stop = False
        self._begin_stop()

    def _restart_server(self) -> None:
        if self._process is None or self._stop_in_progress:
            return
        self._restart_after_stop = True
        self._begin_stop()

    def _begin_stop(self) -> None:
        process = self._process
        if process is None or self._stop_in_progress:
            return
        self._stop_in_progress = True
        self._set_status("Останавливается", YELLOW)
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.restart_button.configure(state="disabled")
        worker = threading.Thread(
            target=self._stop_worker,
            args=(process, self._job_handle),
            name="llama-server-stop",
            daemon=False,
        )
        worker.start()

    def _stop_worker(
        self, process: subprocess.Popen[str], job_handle: int | None
    ) -> None:
        error = ""
        try:
            if process.poll() is None:
                if os.name == "nt":
                    process.terminate()
                    try:
                        process.wait(timeout=2.5)
                    except subprocess.TimeoutExpired:
                        if job_handle:
                            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                            kernel32.TerminateJobObject.argtypes = [
                                ctypes.c_void_p,
                                ctypes.c_uint,
                            ]
                            kernel32.TerminateJobObject(
                                ctypes.c_void_p(job_handle), 1
                            )
                        else:
                            subprocess.run(
                                [
                                    "taskkill",
                                    "/PID",
                                    str(process.pid),
                                    "/T",
                                    "/F",
                                ],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                creationflags=subprocess.CREATE_NO_WINDOW,
                                timeout=5,
                                check=False,
                            )
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=2)
                else:
                    import signal

                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=2.5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=2)
        except Exception as exc:
            error = str(exc)
            try:
                process.kill()
                process.wait(timeout=2)
            except Exception:
                pass
        finally:
            close_windows_job(job_handle)
            self._events.put(("stopped", (process, error)))

    def _poll_process(self) -> None:
        process = self._process
        if process is not None and not self._stop_in_progress:
            return_code = process.poll()
            if return_code is not None:
                close_windows_job(self._job_handle)
                self._job_handle = None
                self._process = None
                self._set_status("Остановлен", RED)
                self.start_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                self.restart_button.configure(state="disabled")
                if return_code == 0:
                    self._append_log(f"[{self._timestamp()}] Сервер завершил работу.")
                else:
                    self._append_log(
                        f"[{self._timestamp()}] Сервер завершился с кодом {return_code}."
                    )
        self.after(200, self._poll_process)

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "log":
                    if payload:
                        self._append_log(payload)
                        if "listening" in payload.lower() or "server is listening" in payload.lower():
                            self._set_status("Работает", GREEN)
                elif kind == "stopped":
                    process, error = payload
                    if self._process is process:
                        self._process = None
                        self._job_handle = None
                    self._stop_in_progress = False
                    self._set_status("Остановлен", RED)
                    if error:
                        self._append_log(f"Ошибка при остановке: {error}")
                    else:
                        self._append_log(f"[{self._timestamp()}] Сервер остановлен.")
                    if self._close_requested:
                        self._save_settings()
                        self.destroy()
                        return
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.restart_button.configure(state="disabled")
                    if self._restart_after_stop:
                        self._restart_after_stop = False
                        self._start_server()
        except queue.Empty:
            pass
        self.after(80, self._drain_events)

    def _set_status(self, text: str, color: str) -> None:
        self.status_dot.configure(text_color=color)
        self.status_text.configure(text=text)

    def _append_log(self, line: str) -> None:
        if not hasattr(self, "log_box"):
            return
        self.log_box.configure(state="normal")
        self.log_box.insert("end", line + "\n")
        try:
            line_count = int(self.log_box.index("end-1c").split(".")[0])
            if line_count > 5000:
                self.log_box.delete("1.0", "1001.0")
        except (ValueError, IndexError):
            pass
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def _update_endpoint(self, *_args: Any) -> None:
        if hasattr(self, "endpoint_label"):
            port = self.port_var.get().strip() or "8080"
            self.endpoint_label.configure(
                text=f"Адрес сервера: http://127.0.0.1:{port}"
            )

    def _on_close(self) -> None:
        self._save_settings()
        if self._process is None:
            self.destroy()
            return
        self._close_requested = True
        self._set_status("Завершение...", YELLOW)
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.restart_button.configure(state="disabled")
        self._begin_stop()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%H:%M:%S")


if __name__ == "__main__":
    LlamaServerLauncher().mainloop()
