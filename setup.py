from setuptools import setup

APP = ["main.py"]
APP_NAME = "RealtimeSTT"

OPTIONS = {
    "argv_emulation": True,
    "packages": ["pyaudio", "websocket", "encodings"],
    "plist": {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
    },
}

setup(
    name="realtime-stt-gui",
    app=APP,
    options={"py2app": OPTIONS},
)
