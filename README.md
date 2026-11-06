# Llama Server Launcher

A Windows GUI for launching and managing a local
[llama.cpp](https://github.com/ggml-org/llama.cpp) server.

## Features

- Select the `llama-server` binary and a GGUF model (file, folder, or Hugging Face cache)
- Configure launch parameters, including port and KV-cache type
- Start, restart and stop the server from one window
- Live server log
- Remembers settings per model
- Child-process cleanup through Windows Job Objects, so no orphan server processes are left behind

## Requirements

- Windows 10 or 11
- Python 3.10+ (when running from source) or the built `.exe`
- `llama-server.exe` and a GGUF model

The app does not download models — point it at files you already have.

## Run from source

```powershell
py -m pip install customtkinter
py llama_server_launcher.py
```

## Build

```powershell
py -m pip install pyinstaller
py -m PyInstaller LlamaServerLauncher.spec
```

The output appears in `dist\`.

## Settings

Settings are stored in `%LOCALAPPDATA%\LlamaServerLauncher\settings.json`.

## License

No license has been added yet.
