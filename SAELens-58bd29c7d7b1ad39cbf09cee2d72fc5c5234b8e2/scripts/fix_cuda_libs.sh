#!/bin/bash
# Fix uv's missing CUDA .so symlinks in the venv.
# Run this after `uv sync` on any machine where the system has
# nvidia packages installed at /usr/local/lib/python3.11/dist-packages/nvidia/
# (standard on most CUDA-enabled cloud VMs).
set -e

SYS=/usr/local/lib/python3.11/dist-packages/nvidia
VENV_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
VENV=$VENV_SITE/nvidia

echo "Linking CUDA libs from $SYS into $VENV"

for lib_dir in cublas cudnn cufft curand cusolver cusparse nccl nvjitlink cuda_nvrtc cuda_cupti cuda_runtime; do
    src="$SYS/$lib_dir/lib"
    dst="$VENV/$lib_dir/lib"
    [ -d "$src" ] || continue
    for f in "$src"/*.so*; do
        [ -f "$f" ] && ln -sf "$f" "$dst/$(basename "$f")" && echo "  linked $lib_dir/$(basename "$f")"
    done
done

echo "Done. Verifying..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
