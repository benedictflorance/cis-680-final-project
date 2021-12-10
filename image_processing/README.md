# Image Processing Pipeline to Count Machine Washer Parts

## Method

[An Image Processing based Object Counting Approach for Machine Vision Application](https://arxiv.org/ftp/arxiv/papers/1802/1802.05911.pdf)

The original authors propose a method of counting the number of objects in images with circular parts, such as soda bottle caps, eggs etc. 

They use the following steps to obtain the circular object count:
1. Convert the input RGB image to HSV color space
2. Isolate the Saturation channel from HSV space
3. Apply a Gaussian filter to the Saturation channel to remove noise and add a blur
4. Perform Otsu thresholding on the image to separate foreground and background
5. Apply Sobel filters to detect edges
6. Apple Hough Circle Transformation on the edge map to generate circles
7. Count the number of generated circles to get the final object count

Since no official implementation of the method was provided, we have written our own code to replicate their experiment in our scenario.

### Quick Start

Instructions for running the whole pipeline are given in the .ipynb notebook itself.

The dataset can be found [here](https://drive.google.com/drive/folders/1OyWLO9ysCCZkGnQdwYhIKbU0ixk_73Zj?usp=sharing)

---



## Citation
Kudos to the authors for their proposed methodology:
```
@misc{baygin2018image,
      title={An Image Processing based Object Counting Approach for Machine Vision Application}, 
      author={Mehmet Baygin and Mehmet Karakose and Alisan Sarimaden and Erhan Akin},
      year={2018},
      eprint={1802.05911},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
