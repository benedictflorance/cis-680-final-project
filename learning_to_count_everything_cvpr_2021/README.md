# Learning To Count Everything (CVPR 2021)

## Method
- Due to large variation in object size: generate the target density map using Gaussian smoothing with adaptive window size.
- First, we use dot annotations to estimate the size of the objects. 
- Given the dot annotation map, where each dot is at an approximate center of an object, we compute the distance between each dot and its nearest neighbor, and average these distances for all the dots in the image. In the paper they have dot annotations for images, in machine-parts dataset we take COM of the polygon masks
- This average distance is used as the size of the Gaussian window to generate the target density map. The standard deviation of the Gaussian is set to be a quarter of the window size
- We follow a similar training methodology as outlined in the paper for our train-val-test split of 80-10-10 (6480-810-810)

## Quick Start

- Run the quickstart tutorial in the `680_Final_Project.ipynb` to run the code.
- Dataset can be found [here](https://drive.google.com/drive/folders/1OyWLO9ysCCZkGnQdwYhIKbU0ixk_73Zj?usp=sharing)
- Scaled dataset can be found [here](https://drive.google.com/drive/folders/1Cus6rW-Rqy2a62qYSL3Pf0DBI4dheGt-?usp=sharing)
- Groundtruth density maps can be found [here](https://drive.google.com/drive/folders/1sC1HYSp8ntKigpAHjMAZZxldXB9YedZT?usp=sharing)
- Our trained model can be found at pretrained/
- data folder can also be accessed [here](https://drive.google.com/drive/folders/1D6Dff_b54SLNxgpS8veooFUV5cxzcMqF?usp=sharing)

## Citation

If you find the code useful, please cite:
```
@inproceedings{m_Ranjan-etal-CVPR21,
  author = {Viresh Ranjan and Udbhav Sharma and Thu Nguyen and Minh Hoai},
  title = {Learning To Count Everything},
  year = {2021},
  booktitle = {Proceedings of the {IEEE/CVF} Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```

