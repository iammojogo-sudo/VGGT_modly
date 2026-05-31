import json
import platform
import subprocess
import sys
from pathlib import Path


IS_WIN = platform.system() == "Windows"


def pip(venv, *args):
    pip_exe = venv / ("Scripts/pip.exe" if IS_WIN else "bin/pip")
    subprocess.run([str(pip_exe)] + list(args), check=True)


def python_exe_in_venv(venv):
    return venv / ("Scripts/python.exe" if IS_WIN else "bin/python")


def setup(python_exe, ext_dir, gpu_sm):
    venv = ext_dir / "venv"

    if not venv.exists():
        print("[setup] Creating venv at %s ..." % venv)
        subprocess.run([str(python_exe), "-m", "venv", str(venv)], check=True)
    else:
        print("[setup] Venv exists, skipping creation.")

    venv_python = python_exe_in_venv(venv)

    try:
        ver = subprocess.run(
            [str(venv_python), "-c",
             "import sys, platform; print(sys.version.split()[0], platform.machine())"],
            capture_output=True, text=True,
        )
        print("[setup] venv Python: %s" % (ver.stdout.strip() or ver.stderr.strip()))
    except Exception:
        pass

    print("[setup] Installing build prerequisites...")
    pip(venv, "install", "setuptools", "wheel")

    # ------------------------------------------------------------------ #
    # PyTorch (CUDA build matched to the GPU)
    # ------------------------------------------------------------------ #
    if gpu_sm >= 100:
        torch_index = "https://download.pytorch.org/whl/cu128"
        torch_pkgs  = ["torch>=2.7.0", "torchvision>=0.22.0"]
        print("[setup] SM %d (Blackwell) -> PyTorch 2.7 + CUDA 12.8" % gpu_sm)
    elif gpu_sm >= 70:
        torch_index = "https://download.pytorch.org/whl/cu124"
        torch_pkgs  = ["torch==2.6.0", "torchvision==0.21.0"]
        print("[setup] SM %d -> PyTorch 2.6.0 + CUDA 12.4" % gpu_sm)
    else:
        torch_index = "https://download.pytorch.org/whl/cu118"
        torch_pkgs  = ["torch==2.5.1", "torchvision==0.20.1"]
        print("[setup] SM %d (legacy) -> PyTorch 2.5.1 + CUDA 11.8" % gpu_sm)

    print("[setup] Installing PyTorch...")
    pip(venv, "install", *torch_pkgs, "--index-url", torch_index)

    # ------------------------------------------------------------------ #
    # Core dependencies — installed individually so one missing wheel can't
    # abort the whole batch, and so a failure names the actual package.
    # ------------------------------------------------------------------ #
    required = ["numpy", "Pillow", "huggingface_hub", "safetensors", "einops", "trimesh"]
    optional = ["scipy", "tqdm", "opencv-python-headless"]

    print("[setup] Installing required dependencies...")
    failed_required = []
    for pkg in required:
        try:
            pip(venv, "install", pkg)
        except subprocess.CalledProcessError:
            failed_required.append(pkg)
            print("[setup] *** REQUIRED package failed: %s ***" % pkg)

    if failed_required:
        raise RuntimeError(
            "Required packages failed to install: %s. See the pip error above — "
            "usually 'no matching distribution' for this Python version."
            % ", ".join(failed_required)
        )

    print("[setup] Installing optional dependencies...")
    for pkg in optional:
        try:
            pip(venv, "install", pkg)
        except subprocess.CalledProcessError:
            print("[setup] Optional package failed (continuing): %s" % pkg)

    # open3d powers ONLY the Poisson 'Mesh' output. RANSAC/Manhattan planar
    # cleanup and point-cloud output are open3d-free, so make it best-effort.
    print("[setup] Installing open3d (optional — only for Poisson mesh output)...")
    try:
        pip(venv, "install", "open3d")
    except subprocess.CalledProcessError:
        print(
            "[setup] open3d could not be installed for this Python version.\n"
            "[setup]   The extension still works: use Output = 'Point cloud only', or\n"
            "[setup]   Planar Cleanup = 'RANSAC planes' / 'Manhattan snap' (all open3d-free).\n"
            "[setup]   Only the Poisson 'Mesh' output is disabled until open3d installs.\n"
            "[setup]   open3d 0.19 ships wheels for CPython up to ~3.13; a 3.10-3.12 venv is safest."
        )

    # ------------------------------------------------------------------ #
    # Clone VGGT repo
    # ------------------------------------------------------------------ #
    repo_dir = ext_dir / "vggt"
    if not repo_dir.exists():
        print("[setup] Cloning VGGT repo...")
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/facebookresearch/vggt.git",
             str(repo_dir)],
            check=True
        )
    else:
        print("[setup] VGGT repo exists, skipping clone.")

    # Install the package WITHOUT deps so it can't override the CUDA torch
    # build above. Its runtime imports (torch/torchvision/numpy/Pillow/einops/
    # huggingface_hub/safetensors) are already installed.
    print("[setup] Installing VGGT package (editable, no-deps)...")
    try:
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-e", str(repo_dir), "--no-deps"],
            check=True
        )
    except subprocess.CalledProcessError:
        print("[setup] 'pip install -e' failed; the extension adds the repo to "
              "sys.path at runtime as a fallback.")

    # ------------------------------------------------------------------ #
    # Verify imports
    # ------------------------------------------------------------------ #
    print("[setup] Verifying VGGT import...")
    check = subprocess.run(
        [str(venv_python), "-c",
         "import sys; sys.path.insert(0, r'%s'); "
         "from vggt.models.vggt import VGGT; "
         "from vggt.utils.geometry import unproject_depth_map_to_point_map; "
         "import trimesh, numpy; print('VGGT: OK')" % str(repo_dir)],
        capture_output=True, text=True,
    )
    if "OK" in check.stdout:
        print("[setup] %s" % check.stdout.strip())
    else:
        print("[setup] Import check FAILED:\n%s"
              % (check.stderr.strip() or check.stdout.strip()))

    o3d = subprocess.run(
        [str(venv_python), "-c", "import open3d; print(open3d.__version__)"],
        capture_output=True, text=True,
    )
    if o3d.returncode == 0:
        print("[setup] open3d %s present — Poisson mesh output enabled." % o3d.stdout.strip())
    else:
        print("[setup] open3d not present — use planar / point-cloud modes (Poisson mesh disabled).")

    print("[setup] Done. Venv ready at: %s" % venv)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(
            python_exe=sys.argv[1],
            ext_dir=Path(sys.argv[2]),
            gpu_sm=int(sys.argv[3]),
        )
    elif len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            python_exe=args["python_exe"],
            ext_dir=Path(args["ext_dir"]),
            gpu_sm=int(args["gpu_sm"]),
        )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm>")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":89}\'')
        sys.exit(1)
