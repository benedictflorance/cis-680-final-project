# Detectron2 (Mask R-CNN for object detection and segmentation)

## Method
- We use the Detectron2 mask rcnn with a FPN backbone.
- At each level of the feature pyramid, a RPN proposes ROIs. With features extracted from the ROIs, an object clasisfication branch outputs a bounding box and object class. For each predicted bounding box and class, another branch generates a corresponding per-pixel mask.
- We follow the same architecure with a post processing aggregation to obtain a washer count. That is, we sum the number of predicted masks.
- We finetune the `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml` model on the washer dataset.
- We follow a similar training methodology as outlined in the paper for our train-val-test split of 80-10-10 (6480-810-810)

## Quick Start
- Install detectron2:
````python
import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/$CUDA_VERSION/torch$TORCH_VERSION/index.html
````
Run `train.py DATA_DIR` to train and 
`test.py DATA_DIR` to test.
- Dataset can be found [here](https://drive.google.com/drive/folders/1OyWLO9ysCCZkGnQdwYhIKbU0ixk_73Zj?usp=sharing)
- Our trained model can be found at pretrained/

## Citation

If you find the detectron2 API useful, please cite:
```
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```

