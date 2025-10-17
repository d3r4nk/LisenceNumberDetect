# LisenceNumberDetect

A Python project that uses K-Nearest Neighbors to detect license plate characters from images. I built this as a lightweight proof-of-concept (and yes, I was lazy to collect a lot of data) 😉  

---

## Overview

LisenceNumberDetect is a simplistic license plate character recognition system. The goal is to take images of license plates, preprocess them, and use a KNN-based classifier to predict the alphanumeric characters on the plate. The project includes tools for data generation, preprocessing, model training, and evaluation.

This is not production-grade—it's intended more as a learning / experimental project.

---

## Features

- **Data generation**: Synthetic generation or augmentation of character images for training.  
- **Preprocessing pipeline**: Steps like thresholding, resizing, flattening, normalization, etc.  
- **KNN classifier**: A simple K-Nearest Neighbors model to classify individual characters.  
- **Character segmentation & flattening**: Extract individual character regions from plate images, flatten to feature vectors.  
- **Evaluation & plotting**: Tools to inspect performance, plot training data distributions, etc.  
- **Test scripts**: Example scripts to run recognition on new images.  

---

## Technologies & Tools Used

- **Python** (core language)  
- Libraries and modules likely (or that could be used):  
  - `numpy` for numerical operations  
  - `scikit-learn` for KNN classifier and model utilities  
  - `OpenCV` (`cv2`) for image processing, thresholding, morphological operations  
  - `matplotlib` or `seaborn` for plotting / visualizations  
- File-based storage of intermediate data: e.g. flattened images, classification labels  
- Project scripts include:  
  - `GenData.py` — generate or collect training data  
  - `Preprocess.py` — image preprocessing routines  
  - `Image_test2.py` — script for testing on new images  
  - `plot.py` — to plot distributions, visuals, etc.  

---

## Getting Started

### Prerequisites

- Python 3.x  
- Install dependencies, e.g.:
  ```bash
  pip install numpy scikit-learn opencv-python matplotlib
