#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_NAME="${APP_NAME:-RealtimeSTT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

export UV_CACHE_DIR

MIC_PERMISSION_DESC="本应用需要访问麦克风用于实时语音转写。"

set_microphone_usage_description() {
    local info_plist="$1"
    if [ ! -f "$info_plist" ]; then
        return 0
    fi

    if /usr/bin/PlistBuddy "$info_plist" -c "Set :NSMicrophoneUsageDescription \"$MIC_PERMISSION_DESC\"" >/dev/null 2>&1; then
        return 0
    fi

    /usr/bin/PlistBuddy "$info_plist" -c "Add :NSMicrophoneUsageDescription string \"$MIC_PERMISSION_DESC\"" >/dev/null 2>&1 || true
}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "未找到 Python 可执行文件: $PYTHON_BIN"
    exit 1
fi

if ! "$PYTHON_BIN" -m PyInstaller --version >/dev/null 2>&1; then
    echo "未检测到 PyInstaller。"
    echo "请先安装："
    echo "  uv pip install pyinstaller"
    exit 1
fi

cd "$ROOT_DIR"

echo "开始用 PyInstaller 打包 macOS .app..."
if [ -f "$ROOT_DIR/RealtimeSTT.spec" ]; then
    "$PYTHON_BIN" -m PyInstaller \
        --noconfirm \
        --clean \
        "$ROOT_DIR/RealtimeSTT.spec"
else
    "$PYTHON_BIN" -m PyInstaller \
        --noconfirm \
        --clean \
        --windowed \
        --name "$APP_NAME" \
        --collect-all encodings \
        --collect-all tkinter \
        main.py
fi

set_microphone_usage_description "$ROOT_DIR/dist/$APP_NAME.app/Contents/Info.plist"

echo "打包完成: dist/$APP_NAME.app"
