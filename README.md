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
