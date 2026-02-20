from setuptools import setup


APP = ["main.py"]
APP_NAME = "RealtimeSTT"

OPTIONS = {
    "argv_emulation": True,
    "packages": ["pyaudio", "websocket"],
    "plist": {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
    },
}


setup(
    app=APP,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
