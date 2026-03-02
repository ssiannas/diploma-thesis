#!/usr/bin/env python3
"""Verify that the full environment is ready for training."""
import subprocess
import sys
from pathlib import Path


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "OK" if ok else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    return ok


def check_python_packages() -> int:
    print("\n=== Python Packages ===")
    failures = 0
    required = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "scipy": "SciPy",
        "tqdm": "tqdm",
        "plyfile": "plyfile",
        "open3d": "Open3D",
        "yaml": "PyYAML",
    }
    for module, name in required.items():
        try:
            mod = __import__(module)
            ver = getattr(mod, "__version__", "")
            if not check(name, True, ver):
                failures += 1
        except ImportError:
            check(name, False, "not installed")
            failures += 1
    return failures


def check_minkowski_engine() -> int:
    print("\n=== MinkowskiEngine ===")
    try:
        import MinkowskiEngine as ME

        return 0 if check("MinkowskiEngine", True, ME.__version__) else 1
    except ImportError as e:
        check("MinkowskiEngine", False, str(e))
        return 1
    except Exception as e:
        check("MinkowskiEngine", False, f"import error: {e}")
        return 1


def check_gpu() -> int:
    print("\n=== GPU ===")
    failures = 0
    try:
        import torch

        avail = torch.cuda.is_available()
        if avail:
            name = torch.cuda.get_device_name(0)
            count = torch.cuda.device_count()
            check("CUDA", True, f"{count} device(s): {name}")
        else:
            check("CUDA", False, "torch.cuda.is_available() == False")
            failures += 1
    except ImportError:
        check("CUDA", False, "PyTorch not installed")
        failures += 1
    return failures


def check_model_forward_pass() -> int:
    print("\n=== Model Forward Pass ===")
    try:
        import MinkowskiEngine as ME
        import torch

        # Ensure project root is importable
        project_root = str(Path(__file__).resolve().parents[2])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from postprocessing.model import SparseUNet

        model = SparseUNet(in_channels=4)
        model.eval()

        # Synthetic input -- needs enough points for 3-level U-Net downsampling
        N = 512
        coords = torch.randint(0, 64, (N, 3))
        batch_idx = torch.zeros(N, 1, dtype=torch.int32)
        coords = torch.cat([batch_idx, coords.int()], dim=1)
        feats = torch.randn(N, 4)
        x = ME.SparseTensor(feats, coords)

        with torch.no_grad():
            out = model(x)

        ok = out.F.shape[1] == 3 and out.F.shape[0] > 0
        return (
            0
            if check("SparseUNet forward", ok, f"output shape {tuple(out.F.shape)}")
            else 1
        )
    except Exception as e:
        check("SparseUNet forward", False, str(e))
        return 1


def check_datasets() -> int:
    print("\n=== Dataset Symlinks ===")
    failures = 0
    datasets = {
        "datasets/8iVFB": "8iVFB (JPEG Pleno)",
        "datasets/pcgcv2_decoded": "PCGCv2 decoded training data",
        "datasets/ModelNet40": "ModelNet40 (optional)",
    }
    for path_str, name in datasets.items():
        p = Path(path_str)
        if p.is_symlink():
            target = p.resolve()
            is_optional = "optional" in name
            if target.is_dir():
                check(name, True, f"-> {target}")
            else:
                check(name, is_optional, f"symlink target missing: {target}")
                if not is_optional:
                    failures += 1
        elif p.is_dir():
            check(name, True, "directory (not symlink)")
        else:
            is_optional = "optional" in name
            check(name, is_optional, "missing" + (" (optional)" if is_optional else ""))
            if not is_optional:
                failures += 1
    return failures


def check_frameworks() -> int:
    print("\n=== Frameworks ===")
    failures = 0

    pcgcv2 = Path("frameworks/PCGCv2")
    if not check("PCGCv2 submodule", pcgcv2.is_dir()):
        failures += 1

    # Check executables
    for exe_name in ["tmc3", "pc_error_d"]:
        exe = pcgcv2 / exe_name
        if exe.is_file():
            import os

            is_exec = os.access(exe, os.X_OK)
            if not check(
                exe_name,
                is_exec,
                "not executable -- run chmod +x" if not is_exec else "",
            ):
                failures += 1
        else:
            check(exe_name, False, f"not found at {exe}")
            failures += 1

    # TMC13 build
    tmc3_built = Path("frameworks/mpeg-pcc-tmc13/build/tmc3")
    check(
        "TMC13 (G-PCC)",
        tmc3_built.exists(),
        "built" if tmc3_built.exists() else "not built (optional)",
    )

    return failures


def check_pcml_package() -> int:
    print("\n=== pcml Package ===")
    try:
        import pcml

        return 0 if check("pcml importable", True) else 1
    except ImportError:
        check("pcml importable", False, "run: pip install -e .")
        return 1


def main():
    print("=" * 60)
    print("Environment Verification")
    print("=" * 60)

    total_failures = 0
    total_failures += check_python_packages()
    total_failures += check_minkowski_engine()
    total_failures += check_gpu()
    total_failures += check_pcml_package()
    total_failures += check_frameworks()
    total_failures += check_datasets()
    total_failures += check_model_forward_pass()

    print("\n" + "=" * 60)
    if total_failures == 0:
        print("All checks passed. Ready for training.")
    else:
        print(f"{total_failures} check(s) failed. See above for details.")
    return 1 if total_failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
