import torch
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.structures import BoxMode
import numpy as np
import os, json, cv2, random, sys
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer
from glob import glob
from tqdm import tqdm

if(len(sys.argv) < 2): 
    print("USAGE: python train.py DATA_DIR")
    sys.exit(1)

DATA_DIR = sys.argv[1]

#DATA_DIR = "/home/edinella/CIS680/cis-680-final-project/mask-rcnn/dataset-split/"

setup_logger()

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("washertrain",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  
    cfg.SOLVER.MAX_ITER = 9000    
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

    return cfg

def get_washer_dicts(img_dir):
    json_file = os.path.join(img_dir, "annotations.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns["images"]):
        record = {}
        
        f = v["file_name"]
        fname = os.path.join(img_dir, f[f.rfind("/")+1:])
        if not os.path.exists(fname): continue

        record["file_name"] = fname
        record["id"] = v["id"]
        record["height"] = v["height"]
        record["width"] = v["width"]
      
        objs = []
        annos = [s for s in imgs_anns["annotations"] if s["image_id"] == v["id"]]
        for anno in annos:
            x = anno["bbox"][0]
            y = anno["bbox"][1]
            w = anno["bbox"][2]
            h = anno["bbox"][3]
            poly = []
            for x_p in range(int(x), int(x+w)):
                for y_p in range(int(y), int(y+h)):
                    poly += [(x_p + .5, y_p + .5)]

            obj = {
                "bbox": anno["bbox"], 
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def main(): 
    for d in ["train", "val", "test"]:
        DatasetCatalog.register("washer" + d, lambda d=d: get_washer_dicts(os.path.join(DATA_DIR, d)))
        MetadataCatalog.get("washer" + d).thing_classes = ["washer"]

    washer_metadata = MetadataCatalog.get("washer_train")

    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    checkpointer = DetectionCheckpointer(predictor.model, save_dir="output")
    checkpointer.save("model")  # save to output/model_999.pth

if __name__ == '__main__':
    main()

