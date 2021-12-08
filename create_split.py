import os
import json
import random
from glob import glob
from collections import defaultdict

OG_DIR = "dataset/"
train_split = .8
val_split = .1

angles = defaultdict(list)
split = {}

for img in glob(os.path.join(OG_DIR, "*_img.png")):
    img = img.replace(OG_DIR, "")
    _id = int(img[0:img.find("_")])

    angles[_id].append(img)

for k, v in angles.items():
    assert len(v) == 9


all_angles = list(angles.keys())
random.shuffle(all_angles)
end_idx = int(len(all_angles)*train_split)
test_idx = end_idx + int(len(all_angles)*val_split)

train_keys = all_angles[0:end_idx]
val_keys = all_angles[end_idx:test_idx]
test_keys = all_angles[test_idx:]

print(len(train_keys), len(val_keys), len(test_keys))

train = []
for k in train_keys:
    train += angles[k] 

val = []
for k in val_keys:
    val += angles[k]

test = []
for k in test_keys:
    test += angles[k]

print(len(train), len(val), len(test))

split["train"] = train
split["test"] = test
split["val"] = val

with open("dataset_split_angles.json", "w") as f:
    json.dump(split, f)
