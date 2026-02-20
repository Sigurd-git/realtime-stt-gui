# realtime-stt-gui

本项目把终端版 `main.py` 改为可直接运行的 Tkinter GUI，并保留 `--cli` 终端模式。

## 运行前准备

```bash
python3 -m pip install websocket-client pyaudio
```

### 用 uv 部署环境

```bash
cd /Users/sigurd/Documents/Projects/realtime-stt-gui
export UV_CACHE_DIR=$(pwd)/.uv-cache
uv venv .venv
source .venv/bin/activate
uv pip install websocket-client pyaudio
```

如果要打包 .app，可继续：

```bash
uv pip install py2app
```

注意：`UV_CACHE_DIR` 用于规避默认缓存目录无权限问题。

macOS 需要麦克风权限，首次运行时请在“系统设置 -> 隐私与安全性 -> 麦克风”允许终端/应用访问。

## 启动方式

### GUI（推荐）

```bash
python3 main.py
```

若无字幕，可先跑 debug 输出：

```bash
python3 main.py --debug
```

GUI 下加上：

```bash
python3 main.py --debug
```

观察 `[raw] ...` 里的 `type` 是否有 `transcript`/`delta` 关键字，按需贴出来我再帮你继续适配。

### 终端模式

```bash
OPENAI_API_KEY=... python3 main.py --cli
```

或

```bash
python3 main.py --cli --api-key YOUR_KEY
```

## 打包为 .app（macOS）

使用 py2app（推荐）：

```bash
python3 -m pip install py2app
python3 setup.py py2app
```

或使用 PyInstaller：

```bash
python3 -m pip install pyinstaller
pyinstaller --windowed --name RealtimeSTT main.py
```

执行后会在 `dist/` 下生成 `RealtimeSTT.app`。
