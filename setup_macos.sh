#!/bin/zsh
set -euo pipefail

repo_dir=${0:A:h}
venv_dir=${TRELLIS_VENV:-"$repo_dir/.venv"}
uv_cache_dir=${UV_CACHE_DIR:-"/private/tmp/trellis-uv-cache"}

if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
  print -u2 "setup_macos.sh requires an Apple Silicon Mac."
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  print -u2 "Install uv first: brew install uv"
  exit 1
fi
cd "$repo_dir"
export UV_CACHE_DIR="$uv_cache_dir"
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-14.0}

if [[ ! -x "$venv_dir/bin/python" ]]; then
  uv venv "$venv_dir" --python 3.11
fi

print "Installing Python and MLX dependencies..."
uv pip install --python "$venv_dir/bin/python" -r requirements_macos.txt

print "Building native Metal extensions with Apple clang..."
metal_packages=(
  "https://github.com/pedronaugusto/mtlgemm/archive/main.tar.gz"
  "https://github.com/pedronaugusto/mtldiffrast/archive/main.tar.gz"
  "https://github.com/pedronaugusto/mtlmesh/archive/main.tar.gz"
  "https://github.com/pedronaugusto/mtlbvh/archive/main.tar.gz"
)
for package in "${metal_packages[@]}"; do
  uv pip install \
    --python "$venv_dir/bin/python" \
    --no-build-isolation \
    "$package"
done

print "Building o-voxel CPU extension (Metal handles raster/BVH work)..."
git submodule update --init --recursive
BUILD_TARGET=cpu uv pip install \
  --python "$venv_dir/bin/python" \
  --no-build-isolation \
  --editable \
  ./o-voxel

print "Validating Hugging Face authentication..."
"$venv_dir/bin/hf" auth whoami
print
print "Setup complete. Download weights with:"
print "  $venv_dir/bin/python scripts/download_weights.py --output-dir weights/TRELLIS.2-4B"
