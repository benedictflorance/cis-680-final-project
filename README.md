# Counting Objects in Cluttered Environments

This project has been implemented as a requirement for the Final Project for the Fall 2021 iteration of the CIS-680 course at the University of Pennsylvania.

We implement a model to count machine parts bunched up together in a cluttered image.

## Installation

```
git clone placeholder.git
cd placeholder
pip install -r requirements.txt
```

<details>
  <summary> Dependencies (click to expand) </summary>
  
  ## Dependencies
  - PyTorch 1.4
  - matplotlib
  - numpy
  - imageio
  - imageio-ffmpeg
  - configargparse
  
</details>

## How To Run?

### Quick Start


---

To train the model: 

```
python train.py
```

---

To test the model: 

```
python test.py
```


### Pre-trained Models

You can download the pre-trained models [here](https://drive.google.com/drive/folders/1jIr8dkvefrQmv737fFm2isiT6tqpbTbv).
```
├── root 
│   ├── dir0
│   ├── dir1
│   ├── dir2
```

### Reproducibility 

Tests that ensure the results of all functions and training loop match the official implentation are contained in a different branch `reproduce`. One can check it out and run the tests:
```
git checkout reproduce
py.test
```

## Method

[Learning To Count Everything](https://github.com/cvlab-stonybrook/LearningToCountEverything)
  

> Basic concept explanation with image
