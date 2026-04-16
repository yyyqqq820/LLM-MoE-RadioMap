import os
import sys
import glob
import json
import numpy as np
import time
import io
import base64
import matplotlib.pyplot as plt
from collections import defaultdict
from openai import OpenAI

# Optional: Set HF endpoint if required in specific regions
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Configuration and mapping
SCENARIO_MAP = {
    'T01': 'Grassland', 'T02': 'Island', 'T03': 'Ocean', 'T04': 'Lake',
    'T05': 'Suburban', 'T06': 'DenseUrban', 'T07': 'Rural', 'T08': 'OrdinaryUrban',
    'T09': 'Desert', 'T10': 'Mountainous', 'T11': 'Forest'
}

FREQ_MAP = {
    'f00': '150 MHz (VHF)', 'f01': '1500 MHz (L-band)', 'f02': '1700 MHz (L-band)',
    'f03': '3500 MHz (Sub-6GHz)', 'f04': '22000 MHz (mmWave)'
}


def parse_tx_info(txt_path):
    tx_dict = {}
    if not os.path.exists(txt_path):
        return tx_dict
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    tx_dict[parts[0]] = json.loads(parts[1])
                except json.JSONDecodeError:
                    continue
    return tx_dict


# Topology render engine
def generate_ground_topology_base64(building_map_z0, tx_positions):
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.imshow(building_map_z0 > 0, cmap='binary', origin='lower')
    for tx in tx_positions:
        x = max(0, min(127, tx['x']))
        y = max(0, min(127, tx['y']))
        ax.plot(x, y, marker='*', color='red', markersize=12)
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# VLM semantic generator
def get_vlm_semantic_generation(client, base64_image, terrain, freq, coverage_ratio, tx_count):
    if coverage_ratio < 0.01:
        logic_instruction = (
            "SCENARIO A (Absolute Free Space): The coverage is effectively 0.0%. "
            "Focus exclusively on ideal LoS (Line-of-Sight) and spherical wave expansion. "
            "Describe the total absence of structural multipath or shadowing."
        )
    elif 0.01 <= coverage_ratio < 5.0:
        logic_instruction = (
            f"SCENARIO B (Sparse Scattering): The coverage is very low ({coverage_ratio:.2f}%). "
            "Buildings act as 'Isolated 10m Pillars'. Focus on individual knife-edge diffraction "
            "and why the overall environment remains LoS-dominant despite sparse 10m-thick obstacles."
        )
    else:
        logic_instruction = (
            f"SCENARIO C (Urban Clutter): The coverage is significant ({coverage_ratio:.1f}%). "
            "Analyze the 'Contiguous Urban Massifs'. Focus on 10m-wide waveguide corridors, "
            "multiple-reflection paths, and severe shadow fading behind massive 10m+ thick structures."
        )

    system_prompt = (
        "You are an AI physics-engine profiler for electromagnetic wave propagation. "
        "Analyze the provided 2D ground-level spatial topology (1 pixel = 10 meters). "
        "OUTPUT RULE: Write exactly ONE cohesive, high-density scientific diagnosis paragraph (80-120 words). "
        "NO headings, NO bullet points, NO brackets. Be diverse and case-specific."
    )

    prompt = (
        f"Input: {terrain} | {freq} | {coverage_ratio:.2f}% coverage | {tx_count} Tx.\n"
        f"Scale: 1.28km x 1.28km grid (10m/pixel). Receiver: 1.5m high.\n\n"
        f"MANDATORY START: Start by stating: 'In this {terrain} environment with {coverage_ratio:.2f}% building coverage and {tx_count} transmitters operating at {freq}...'\n\n"
        f"Task Instruction: {logic_instruction}\n\n"
        f"Final Goal: Explain the physical reasoning for the power distribution. "
        f"Specifically describe how {freq} interacts with these 10m-quantized structures."
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]}
                ],
                temperature=0.25,
                max_tokens=400
            )
            full_text = response.choices[0].message.content.strip()
            full_text = full_text.replace("```json", "").replace("```", "").replace("\n", " ").strip()

            words = full_text.split()
            if len(words) > 180:
                full_text = " ".join(words[:180])
            return full_text

        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2.0)
            else:
                return (
                    f"In this {terrain} environment with {coverage_ratio:.1f}% coverage and {tx_count} transmitters operating at {freq}, "
                    f"the 10m-quantized spatial layout dictates propagation via frequency-specific mechanisms."
                )


# Main controller
def main():
    root_dir = "./SpectrumNet"
    tx_info_path = os.path.join(root_dir, 'tx_info.txt')
    npz_dir = os.path.join(root_dir, 'npz')
    png_dir = os.path.join(root_dir, 'png')
    out_path = os.path.join(root_dir, "vlm_semantics.json")

    api_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
    # Uncomment the base_url if you are accessing from a restricted region
    client = OpenAI(api_key=api_key)
    # client = OpenAI(api_key=api_key, base_url="https://api.openai-proxy.org/v1")

    tx_data = parse_tx_info(tx_info_path)
    npz_files = glob.glob(os.path.join(npz_dir, "**", "*_bdtr.npz"), recursive=True)
    npz_index = {os.path.basename(p).replace("_bdtr.npz", ""): p for p in npz_files}
    png_files = glob.glob(os.path.join(png_dir, "**", "*.png"), recursive=True)

    groups = defaultdict(set)
    for p in png_files:
        name = os.path.basename(p)
        if '_f' in name:
            tx_key = name.split('_f')[0]
            f_code = name[name.find('_f') + 1: name.find('_f') + 4]
            if tx_key in tx_data and tx_key in npz_index:
                groups[tx_key].add(f_code)

    valid_tx_keys = [k for k, f in groups.items() if len(f) == 5]

    pec_dict = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding='utf-8') as f:
            pec_dict = json.load(f)

    generated_in_this_run = 0

    for tx_key in valid_tx_keys:
        try:
            npz = np.load(npz_index[tx_key])
            bldg = npz['inBldg_zyx'][0].astype(np.float32)
            cov = float((bldg > 0).mean() * 100)
            tx_pos = tx_data[tx_key]
        except Exception:
            continue

        t_desc = SCENARIO_MAP.get(tx_key[:3], "Unknown Terrain")
        base64_img = generate_ground_topology_base64(bldg, tx_pos)

        for f_code, freq_desc in FREQ_MAP.items():
            dict_key = f"{tx_key}_{f_code}"
            if dict_key in pec_dict:
                continue

            llm_summary = get_vlm_semantic_generation(client, base64_img, t_desc, freq_desc, cov, len(tx_pos))
            pec_dict[dict_key] = llm_summary
            generated_in_this_run += 1

            if generated_in_this_run % 10 == 0:
                with open(out_path, "w", encoding='utf-8') as f:
                    json.dump(pec_dict, f, indent=4)
            time.sleep(0.4)

    with open(out_path, "w", encoding='utf-8') as f:
        json.dump(pec_dict, f, indent=4)


if __name__ == "__main__":
    main()