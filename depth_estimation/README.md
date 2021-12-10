# Depth Estimation using Vision Transformers for Dense Prediction

Since the given dataset consists of cluttered objects, we decided to explore the method of Depth Estimation to better estimate the approximate position of an object in the image.

## Method
[Vision Transformers for Dense Prediction](https://github.com/isl-org/DPT).
> René Ranftl, Alexey Bochkovskiy, Vladlen Koltun

The DPT model is the implementation of the [paper](https://arxiv.org/abs/2103.13413) of the same title as above. 
Due to it's superior results in the task of Depth Estimation without the need for any training on a dataset, we use the DPT model for our experiments.

## Quick Start

The provided dpt.ipynb has instructions on how to run the code and obtain the corresponding depth maps.


## Citation

Please cite the following papers if implementing the DPT model.
```
@article{Ranftl2021,
	author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
	title     = {Vision Transformers for Dense Prediction},
	journal   = {ArXiv preprint},
	year      = {2021},
}
```

```
@article{Ranftl2020,
	author    = {Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
	journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
	year      = {2020},
}
```
