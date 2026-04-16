import os
import glob
import json
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange


# =========================================================================
# Core Physical Calculation
# =========================================================================
@jit(nopython=True, parallel=True)
def calculate_single_depth_dis_map_numba(building_map, tx_pos, f_k, beta, alpha):
    rows, cols = building_map.shape
    depth_map = np.zeros((rows, cols), dtype=np.float32)

    log_fk = beta * np.log10(f_k)
    tx_col, tx_row = tx_pos

# Global bias to ensure positive feature values
    C = 150.0

    for i in prange(rows):
        for j in range(cols):
            d = np.sqrt((i - tx_row) ** 2 + (j - tx_col) ** 2)
            d_km = d * 10.0 / 1000.0

# Prevent division by zero or extreme values at TX center
            d_km = max(0.005, d_km)
            log_d = alpha * np.log10(d_km)

            x0, y0 = tx_row, tx_col
            x1, y1 = i, j
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy
            total_pts = max(dx, dy) + 1
            free_pts = 0

            x, y = x0, y0
            for _ in range(total_pts):
                if 0 <= x < rows and 0 <= y < cols and building_map[x, y] == 0:
                    free_pts += 1
                if x == x1 and y == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy

            T = free_pts / total_pts if total_pts > 0 else 0.0

            val = T * (C - log_fk - log_d)
            depth_map[i, j] = max(0.0, val)

    return depth_map

def calculate_multi_depth_map_optimized(building_map, tx_positions, f_k, beta=20.0, alpha=35.0):
    combined_depth = np.zeros_like(building_map, dtype=np.float32)
    for tx_pos in tx_positions:
        single_depth = calculate_single_depth_dis_map_numba(building_map, tx_pos, f_k, beta, alpha)
        # Use maximum signal strength (minimum attenuation) for multi-TX scenarios
        combined_depth = np.maximum(combined_depth, single_depth)
    return combined_depth

def parse_tx_info(txt_path):
    tx_dict = {}
    if not os.path.exists(txt_path):
        return tx_dict
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    tx_data = json.loads(parts[1])
                    positions = []
                    for tx in tx_data:
                        x = min(max(int(round(tx['x'])), 0), 127)
                        y = min(max(int(round(tx['y'])), 0), 127)
                        positions.append((x, y))
                    tx_dict[parts[0]] = positions
                except json.JSONDecodeError:
                    continue
    return tx_dict


def main():
    parser = argparse.ArgumentParser(description="Generate Physical Depth Maps for SpectrumNet")
    parser.add_argument("--data_root", type=str, default="./SpectrumNet", help="Root directory of SpectrumNet dataset")
    parser.add_argument("--max_viz", type=int, default=50, help="Maximum number of visualization images to generate")
    args = parser.parse_args()

    root_dir = args.data_root
    tx_info_path = os.path.join(root_dir, 'tx_info.txt')
    npz_dir = os.path.join(root_dir, 'npz')

    output_png_dir = os.path.join(root_dir, 'depth_images_v2')
    output_npy_dir = os.path.join(root_dir, 'depth_data_v2')
    os.makedirs(output_png_dir, exist_ok=True)
    os.makedirs(output_npy_dir, exist_ok=True)

    FREQ_MAP = {
        'f00': 150.0, 'f01': 1500.0, 'f02': 1700.0, 'f03': 3500.0, 'f04': 22000.0
    }

    beta, alpha = 20.0, 35.0
    tx_data = parse_tx_info(tx_info_path)
    npz_files = glob.glob(os.path.join(npz_dir, "**", "*_bdtr.npz"), recursive=True)
    random.shuffle(npz_files)

    scene_count, skip_no_tx, skip_error, total_generated = 0, 0, 0, 0
    start_time = time.time()

    for npz_path in npz_files:
        filename = os.path.basename(npz_path)
        tx_key = filename.replace("_bdtr.npz", "")

        if tx_key not in tx_data or not tx_data[tx_key]:
            skip_no_tx += 1
            continue

        try:
            npz = np.load(npz_path)
            building_map = npz['inBldg_zyx'][0].astype(np.float32)
        except Exception:
            skip_error += 1
            continue

        tx_positions = tx_data[tx_key]

        for f_key, f_MHz in FREQ_MAP.items():
            depth_map = calculate_multi_depth_map_optimized(
                building_map, tx_positions, f_k=f_MHz, beta=beta, alpha=alpha
            )

            save_base_name = f"{tx_key}_{f_key}_ss_z00"
            npy_save_path = os.path.join(output_npy_dir, f"{save_base_name}.npy")
            np.save(npy_save_path, depth_map)
            total_generated += 1

 # Generate visualizations for a limited subset
            if scene_count < args.max_viz and f_key == 'f00':
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(building_map, cmap='gray', origin='lower')
                plt.title(f"Building Map ({tx_key})")

                plt.subplot(1, 2, 2)
                plt.imshow(depth_map, cmap='viridis', origin='lower')
                for (tx_x, tx_y) in tx_positions:
                    plt.scatter(tx_x, tx_y, color='red', marker='*', s=150, edgecolors='white', linewidths=0.5)

                plt.title(f"Depth Map ({f_MHz} MHz)")
                plt.colorbar(label='Spatial Feature')

                plt.tight_layout()
                png_save_path = os.path.join(output_png_dir, f"{save_base_name}.png")
                plt.savefig(png_save_path, dpi=150)
                plt.close()

        scene_count += 1

    print("-" * 50)
    print("[INFO] Generation Completed")
    print(f"Processed scenes: {scene_count} | Generated files: {total_generated}")
    print(f"Elapsed time: {time.time() - start_time:.2f}s")
    print("-" * 50)


if __name__ == "__main__":
    main()