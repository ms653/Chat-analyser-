#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Build script — creates "WhatsApp Analyser.app" in the dist/ folder
# Run once from this directory:  chmod +x build.sh && ./build.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "Installing build dependencies…"
pip3 install pyinstaller pywebview --quiet

echo "Building .app…"
pyinstaller \
  --windowed \
  --onedir \
  --name "WhatsApp Analyser" \
  --hidden-import "webview" \
  --hidden-import "webview.platforms.cocoa" \
  --hidden-import "clr_loader" \
  --collect-all "webview" \
  --noconfirm \
  gui.py

echo ""
echo "Done!  Your app is at:  dist/WhatsApp Analyser.app"
echo "Drag it to /Applications or double-click from Finder."
