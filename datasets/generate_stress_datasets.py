"""
Generates all SutraOpt stress test datasets including large-scale (1k, 5k, 10k variable) models.
Run: python datasets/generate_stress_datasets.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sutraopt.linalg.gpu_acceleration import MassiveScaleStressGenerator

os.makedirs("datasets", exist_ok=True)

configs = [
    {"num_vars": 1000,  "num_constraints": 200,  "density": 0.01,  "seed": 1},
    {"num_vars": 5000,  "num_constraints": 1000, "density": 0.005, "seed": 2},
    {"num_vars": 10000, "num_constraints": 2000, "density": 0.005, "seed": 3},
]

for cfg in configs:
    n = cfg["num_vars"]
    print(f"Generating {n}-variable stress model...")
    data = MassiveScaleStressGenerator.generate_large_scale_generic_lp(**cfg)
    fname = f"datasets/massive_stress_{n}var.json"
    with open(fname, "w") as f:
        json.dump(data, f)
    size_mb = os.path.getsize(fname) / 1024 / 1024
    print(f"  Saved {fname} ({size_mb:.1f} MB)")

# Also generate mega refinery model
print("Generating mega refinery model (40 crudes x 15 products)...")
refinery = MassiveScaleStressGenerator.generate_mega_refinery_complex(40, 15)
with open("datasets/mega_refinery_40crude_15product.json", "w") as f:
    json.dump(refinery, f, indent=2)

print("\nAll stress datasets generated successfully.")
