#!/data/data/com.termux/files/usr/bin/bash
# One-time setup of studiomix on Android with Termux (install Termux from F-Droid, not Play Store).
#   bash scripts/termux-setup.sh
set -e

echo "==> updating packages"
pkg update -y

echo "==> installing python, numpy, scipy and ffmpeg from Termux packages (no compiling)"
pkg install -y python python-numpy ffmpeg git
if ! pkg install -y python-scipy; then
    # older Termux mirrors ship scipy in the TUR repository
    pkg install -y tur-repo
    pkg install -y python-scipy
fi

echo "==> installing studiomix"
cd "$(dirname "$0")/.."
pip install pyloudnorm
pip install -e .

echo "==> giving Termux access to your phone storage (accept the popup)"
termux-setup-storage || true

echo
echo "Done. Example:"
echo "  studiomix ~/storage/downloads/vocal.wav ~/storage/downloads/beat.wav \\"
echo "      -a ~/storage/downloads/adlibs.wav -p trap -o ~/storage/downloads/master"
