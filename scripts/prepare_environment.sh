#!/usr/bin/env bash
# Run only in a reserved CPU allocation with verified persistent project storage.
set -euo pipefail
: "${BIR_PROJECT_ROOT:?Set verified private project storage}"
: "${SLURM_JOB_ID:?Use a scheduled allocation}"
[[ ${SLURM_RESTART_COUNT:-0} == 0 ]] || exit 75
export UV_CACHE_DIR="$BIR_PROJECT_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$BIR_PROJECT_ROOT/python"
export HF_HOME="$BIR_PROJECT_ROOT/cache/huggingface"
export FLORA_MODEL_CACHE="$BIR_PROJECT_ROOT/models"
export PIP_CACHE_DIR="$BIR_PROJECT_ROOT/cache/pip"
export XDG_CACHE_HOME="$BIR_PROJECT_ROOT/cache"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=4
mkdir -p "$BIR_PROJECT_ROOT/tools" "$BIR_PROJECT_ROOT/ops"
python3 - "$BIR_PROJECT_ROOT/tools" <<'PY'
import hashlib, io, json, pathlib, sys, urllib.request, zipfile
root=pathlib.Path(sys.argv[1])
with urllib.request.urlopen('https://pypi.org/pypi/uv/0.9.5/json', timeout=60) as r:
    meta=json.load(r)
wheel=next(x for x in meta['urls'] if 'manylinux_2_17_x86_64' in x['filename'] and x['filename'].endswith('.whl'))
with urllib.request.urlopen(wheel['url'], timeout=120) as r:
    data=r.read()
assert hashlib.sha256(data).hexdigest()==wheel['digests']['sha256']
with zipfile.ZipFile(io.BytesIO(data)) as z:
    member=next(x for x in z.namelist() if x.endswith('/uv') and '.data/scripts/' in x)
    target=root/'uv'; target.write_bytes(z.read(member)); target.chmod(0o700)
PY
UV="$BIR_PROJECT_ROOT/tools/uv"
"$UV" python install 3.12.12
"$UV" venv --python 3.12.12 "$BIR_PROJECT_ROOT/venv"
"$UV" pip install --python "$BIR_PROJECT_ROOT/venv/bin/python" torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
"$UV" pip install --python "$BIR_PROJECT_ROOT/venv/bin/python" -r "$BIR_PROJECT_ROOT/code/configs/requirements-native.txt" pyarrow==21.0.0 pytest==8.4.2
"$UV" pip install --python "$BIR_PROJECT_ROOT/venv/bin/python" --no-deps -e "$BIR_PROJECT_ROOT/code"
if [[ ! -d "$BIR_PROJECT_ROOT/Flora/.git" ]]; then
  git clone --no-checkout https://github.com/NVIDIA/Flora.git "$BIR_PROJECT_ROOT/Flora"
fi
git -C "$BIR_PROJECT_ROOT/Flora" checkout --detach 129a2ada74c526ee4692f3d1b111bea6ca19d5b7
"$BIR_PROJECT_ROOT/venv/bin/python" "$BIR_PROJECT_ROOT/code/scripts/prepare_assets.py"
"$UV" pip freeze --python "$BIR_PROJECT_ROOT/venv/bin/python" > "$BIR_PROJECT_ROOT/ops/environment.freeze.txt"
du -sb "$BIR_PROJECT_ROOT" > "$BIR_PROJECT_ROOT/ops/storage.bytes.txt"
