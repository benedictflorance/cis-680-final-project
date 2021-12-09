import torch
import os, json, cv2, random, argparse
from glob import glob
from tqdm import tqdm

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.structures import BoxMode
from detectron2.checkpoint import DetectionCheckpointer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode

DATA_DIR = "/home/edinella/CIS680/cis-680-final-project/mask-rcnn/dataset-split/"

setup_logger()

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("washertrain",)
    cfg.DATASETS.TEST = ("washertest",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.STEPS = []        
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (washer)

    cfg.MODEL.WEIGHTS = os.path.join("output", "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold

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

def get_label(_id, imgs_anns):
    annos = [s for s in imgs_anns["annotations"] if s["image_id"] == _id]
    return len(annos)


def visualize_predictions(predictor, metadata):
    #randomly select test samples to visulaize
    dataset_dicts = get_washer_dicts(os.path.join(DATA_DIR, "test"))
    for d in random.sample(dataset_dicts, 3):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  
        v = Visualizer(im[:, :, ::-1],
                       metadata=metadata, 
                       scale=0.5
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        outname = os.path.join("preds", d["file_name"][d["file_name"].rfind("/")+1:])
        print("Output for prediction", outname, ": ", len(outputs["instances"]))
        cv2.imwrite(outname, out.get_image())


def calculate_metrics(predictor, test_loader):
    json_file = os.path.join(DATA_DIR, "annotations.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)


    MAE, MSE, total = 0, 0, 0
    
    for sample in tqdm(test_loader):
        output = predictor(cv2.imread(sample[0]["file_name"]))
        count = len(output["instances"])
        label = get_label(sample[0]["id"], imgs_anns)

        MAE += abs(count - label)
        MSE += (label - count)**2
        total += 1

    return MAE / total, MSE / total


def main():
    for d in ["train", "val", "test"]:
        DatasetCatalog.register("washer" + d, lambda d=d: get_washer_dicts(os.path.join(DATA_DIR, d)))
        MetadataCatalog.get("washer" + d).thing_classes = ["washer"]

    washer_metadata = MetadataCatalog.get("washer_train")
    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)
    test_loader = build_detection_test_loader(cfg, "washertest")

    visualize_predictions(predictor, washer_metadata)

    MAE, MSE = calculate_metrics(predictor, test_loader)
    print("Mean average error: {:2.4f}".format(MAE))
    print("Mean squared error: {:2.4f})".format(MSE))


if __name__ == '__main__':
    main()

