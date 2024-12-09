# Bollywood_face_similarity

**`Overview:`**

A transfer learning project using ResNet CNN to match an uploaded face with the most likely Bollywood celebrity.
Project Structure

**`data.plot`**: Visualizes data distribution (number of images per actor/actress)

**`feature_extractor.py`**: Extracts features from the Bollywood celebrities dataset

**`train_test.py:`** Evaluates model accuracy

**`app.py`**: Streamlit web application for face matching

**`Face Detection Method:`**

Multi-task Cascaded Convolutional Neural Network (MTCNN) is used for face detection.

Important Note: 

The current implementation assumes the first detected face array represents the face of interest. Incorrect face selection may lead to inaccurate results.

**`Requirements`**

Python

Required libraries (detailed in each script)

Streamlit

**`Setup and Installation`**

Create a Streamlit environment (recommended using Anaconda)

Install required dependencies

Run app.py in the Streamlit environment

**`Potential Limitations`**

1. Small dataset size
2. Uneven dataset distribution
3. Reliance on pre-trained model
4. Assumption of first face array representing the target face

**`Recommended Improvements`**

1. Expand dataset
2. Balance dataset across celebrities
3. Implement more robust face selection mechanism
4. Fine-tune pre-trained model

**`Contributing`**

Contributions to improve face matching accuracy and expand the dataset are welcome.
