#!/bin/bash
# Source from a PBS train/eval job after entering PBS_O_WORKDIR.

python_module="${RECHARGE_PYTHON_MODULE:-python/3.11.3}"
required_python="${RECHARGE_PYTHON_VERSION:-3.11}"
venv_dir="${RECHARGE_VENV_DIR:-/srv/scratch/$USER/environments/recharge-return-py311}"

module load "$python_module"
if [[ ! -x "$venv_dir/bin/python" ]]; then
    echo "Missing recharge venv: $venv_dir" >&2
    echo "Submit scripts/katana_recharge_setup.pbs first." >&2
    return 1
fi
source "$venv_dir/bin/activate"
python - "$required_python" "$venv_dir" <<'PY'
import sys
from pathlib import Path
import minigrid, stable_baselines3, torch

expected = tuple(map(int, sys.argv[1].split(".")))
if sys.version_info[:2] != expected:
    raise SystemExit(f"Wrong Python version: {sys.version_info[:2]} != {expected}")
if Path(sys.prefix).resolve() != Path(sys.argv[2]).resolve():
    raise SystemExit(f"Wrong active venv: {sys.prefix}")
if minigrid.__version__ != "2.5.0":
    raise SystemExit(f"Wrong MiniGrid version: {minigrid.__version__}")
print(f"Using {sys.executable}; MiniGrid {minigrid.__version__}; SB3 {stable_baselines3.__version__}; PyTorch {torch.__version__}")
PY

export PYTHONPATH="${PBS_O_WORKDIR:?}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
job_tmp="${TMPDIR:-/tmp}"
export MPLCONFIGDIR="$job_tmp/recharge-matplotlib"
export XDG_CACHE_HOME="$job_tmp/recharge-cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
