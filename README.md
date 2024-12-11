# Bollywood Face Similarity

## Overview

A PyTorch-based transfer learning project utilizing ResNet CNN to match an uploaded face with the most likely Bollywood celebrity.

Dataset: [Bollywood Celeb Localized Face Dataset](https://www.kaggle.com/datasets/sushilyadav1998/bollywood-celeb-localized-face-dataset)

### Steps to Get Started
1. **Download the Dataset:** Download the dataset from the link above and place it in the same directory as the project code files.
2. **Clean the Dataset:** Manually remove any unnecessary or poor-quality images from each folder to improve model performance.
3. **Reorganize Images:** Move all images from the subfolders into a single main folder within the dataset.


## Project Structure

- **`data.plot`**: Script for visualizing data distribution (e.g., number of images per actor/actress).
- **`feature_extractor.py`**: Handles feature extraction from the Bollywood celebrity images.
- **`train_test.py`**: Evaluates model performance and accuracy.
- **`app.py`**: Streamlit web application for face matching.


## Face Detection Method

The project uses Multi-task Cascaded Convolutional Neural Network (MTCNN) for face detection. 

**Important Note:** The implementation assumes the first detected face corresponds to the face of interest. If the detection is incorrect, the matching results may be inaccurate.


## Requirements

- Python (3.x recommended)
- Libraries detailed in each script
- Streamlit for the web application

## Setup and Installation

- Create a Streamlit environment (recommended using Anaconda)
- Install required dependencies
- Run app.py in the Streamlit environment

## Potential Limitations

1. Small dataset size
2. Uneven dataset distribution
3. Reliance on pre-trained model
4. Assumption of first face array representing the target face

## Recommended Improvements

1. Expand dataset
2. Balance dataset across celebrities
3. Implement more robust face selection mechanism
4. Fine-tune pre-trained model

## Contributing

Contributions to improve face matching accuracy and expand the dataset are welcome.
