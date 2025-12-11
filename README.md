# Automatic Document Skew Correction

This project corrects rotated document images using classical computer vision methods.  
It estimates the dominant angle of a document page and rotates the image back to an upright position.

## Course

CS 4337 – Computer Vision  
Texas State University

## Project overview

Many scanned or photographed documents appear tilted.  
This project:

- Detects the dominant angle of the document using edge detection and the Hough transform.
- Rotates the page to make it horizontal.
- Evaluates accuracy on a synthetic dataset with known rotation angles.

## Repository structure
```text
.
├─ src/
│  ├─ deskew.py         # Core pipeline: angle estimation and rotation
│  ├─ evaluation.py     # Computes angle error metrics on synthetic dataset
│  └─ main.py           # Command line demo for a single image
├─ data/
│  └─ synthetic/
│     ├─ *.jpg          # Synthetic document images
│     └─ annotations.csv# Ground truth angles for each image
├─ report/
│  └─ Automatic Document Skew Correction.pdf
├─ requirements.txt
└─ README.md
```

## Setup

You need Python 3.9+.
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` should contain at least:
```text
numpy
opencv-python
```

Add `matplotlib` if you used it.

## How to run the demo

Deskew a single image:
```bash
python src/main.py data/synthetic/angry_rot_20.jpg
```

This prints the estimated angle and saves a corrected image as:
```text
data/synthetic/angry_rot_20_deskewed.jpg
```

You can run this on any image in `data/synthetic`.

## How to run the evaluation

To evaluate the system on the synthetic dataset of 100 images:
```bash
python src/evaluation.py
```

You should see output similar to:
```text
Number of images: 100
Mean absolute error (deg): 0.490
Median absolute error (deg): 0.500
Max absolute error (deg): 0.500
Within 1 degree: 100.0%
Within 2 degrees: 100.0%
```

These numbers are used in the report.

## Data

The synthetic dataset is generated from 10 base images. Each base image is rotated by angles from -20 to +20 degrees in steps of 5 degrees. The `annotations.csv` file stores the mapping from image filename to true rotation angle in degrees.

Example rows:
```csv
image,true_angle_deg
angry_true.jpg,0
angry_rot_-20.jpg,-20
angry_rot_0.jpg,0
angry_rot_20.jpg,20
```

## Report

The full project report, including figures and discussion, is in:
```text
report/Automatic Document Skew Correction.pdf
```