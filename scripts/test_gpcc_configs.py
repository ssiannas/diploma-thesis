"""Test G-PCC with different configuration files."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from pcml.data.loaders import PLYPointCloudLoader
from pcml.frameworks.gpcc import GPCCAdapter
from pcml.metrics.quality import GeometryQualityCalculator


def test_config(input_ply: str, config_name: str, output_dir: Path):
    """Test G-PCC with specific config."""
    print(f"\n{'='*60}")
    print(f"Testing G-PCC with config: {config_name}")
    print(f"{'='*60}")

    loader = PLYPointCloudLoader(verbose=False)
    pc = loader.load(input_ply)

    print(f"Input points: {pc.num_points:,}")

    adapter = GPCCAdapter(config_name=config_name)
    compressed_data, result = adapter.compress(pc.geometry, pc.colors)

    print(f"Compressed size: {result.compressed_size_bytes:,} bytes")
    print(f"Compression time: {result.compression_time_seconds:.3f}s")

    recon_geometry, recon_colors = adapter.decompress(compressed_data)

    metrics = GeometryQualityCalculator.calculate_all(pc.geometry, recon_geometry)
    bpp = (result.compressed_size_bytes * 8) / pc.num_points

    print(f"BPP: {bpp:.2f}")
    print(f"PSNR: {metrics.psnr:.2f} dB")
    print(f"MSE: {metrics.mse:.6f}")

    config_dir = output_dir / config_name.replace(".cfg", "")
    config_dir.mkdir(parents=True, exist_ok=True)

    with open(config_dir / "metrics.txt", "w") as f:
        f.write(f"G-PCC Configuration: {config_name}\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Points: {pc.num_points:,}\n")
        f.write(f"Compressed size: {result.compressed_size_bytes:,} bytes\n")
        f.write(f"BPP: {bpp:.2f}\n")
        f.write(f"PSNR: {metrics.psnr:.2f} dB\n")
        f.write(f"MSE: {metrics.mse:.6f}\n")
        f.write(f"Compression time: {result.compression_time_seconds:.3f}s\n")

    return {
        "config": config_name,
        "bpp": bpp,
        "psnr": metrics.psnr,
        "mse": metrics.mse,
        "compressed_size": result.compressed_size_bytes,
    }


def main():
    input_ply = "datasets/8iVFB_small/longdress_vox10_1300.ply"
    output_dir = Path("results/gpcc_configs")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_dir = Path("frameworks/mpeg-pcc-tmc13/cfg")
    configs = [
        "octree-predlift.cfg",
        "octree-liftt-ctc-lossless-geom-lossy-attrs.cfg",
        "octree-liftt-ctc-lossy-geom-lossy-attrs.cfg",
    ]

    available_configs = []
    for cfg in configs:
        if (cfg_dir / cfg).exists():
            available_configs.append(cfg)
        else:
            print(f"Warning: Config not found: {cfg}")

    if not available_configs:
        print("No configs found! Using default.")
        available_configs = [None]

    results = []
    for config in available_configs:
        result = test_config(input_ply, config or "default", output_dir)
        results.append(result)

    print(f"\n{'='*60}")
    print("Summary of All Configs")
    print(f"{'='*60}")
    print(f"{'Config':<40} {'BPP':>8} {'PSNR':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['config']:<40} {r['bpp']:>8.2f} {r['psnr']:>8.2f}")

    with open(output_dir / "summary.txt", "w") as f:
        f.write("G-PCC Configuration Comparison\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Config':<40} {'BPP':>8} {'PSNR':>8}\n")
        f.write("-" * 60 + "\n")
        for r in results:
            f.write(f"{r['config']:<40} {r['bpp']:>8.2f} {r['psnr']:>8.2f}\n")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
