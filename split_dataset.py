import json
import os
import shutil

OG_DIR = "dataset/"
OUT_DIR = "dataset-split/"

split = json.load(open("dataset_split.json"))

if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

for k, v in split.items():
    print(k, len(v))
    print(v[0])
    split_dir = os.path.join(OUT_DIR, k)
    if not os.path.exists(split_dir):
        os.mkdir(split_dir)

    for sample in v:
        print("copying", sample, "to", split_dir)
        shutil.copy(os.path.join(OG_DIR, sample), split_dir)
        
