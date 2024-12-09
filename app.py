import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from facenet_pytorch import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T
from PIL import Image
import streamlit as st
import torch.nn as nn

feature_list = np.array(pickle.load(open('/Users/niramaypatel/Desktop/python_DSE/bollywood_face_similarity/embedding.pkl', 'rb')))
filenames = pickle.load(open('/Users/niramaypatel/Desktop/python_DSE/bollywood_face_similarity/filenames.pkl', 'rb'))

preprocess = T.Compose([
    T.ToTensor(),
    T.Resize(224),
    T.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
   )
])

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        return self.model(x)
     
feature_extractor = Model()
detector = MTCNN()

# Code for training the model on image dataset

# features_lst=[]
# filenames = pickle.load(open('filenames.pkl','rb'))
# for filename in filenames:
#     image = Image.open(filename).convert('RGB')
#     input_tensor = preprocess(image).unsqueeze(0)
#     with torch.no_grad():
#         features = model(input_tensor)
#     features_lst.append(features.flatten())


# store the feature list in a file
# pickle.dump(features_lst,open("embedding.pkl","wb"))

def save_uploaded_image(uploaded_image):
    try:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        
        image_path = os.path.join('uploads', uploaded_image.name)
        with open(image_path, 'wb') as f:
            f.write(uploaded_image.getbuffer())
        
        return image_path  
    except Exception as e:
        st.error(f"Failed to save the image: {e}")
        return None

def extract_features(sample_img_path,feature_extractor,detector):
    # Face detection
    sample_img = cv2.imread(sample_img_path)
    results = detector.detect(sample_img)
    
    if results[0] is not None:
        boxes = results[0]
        x_min, y_min, x_max, y_max = map(int, boxes[0])  # Assuming the first face
        sample_img = sample_img[y_min:y_max, x_min:x_max]
    else:
        raise ValueError("No face detected in the image.")
    
    face_image = Image.fromarray(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(face_image).unsqueeze(0)
    
    # Feature extraction
    with torch.no_grad():
        features = feature_extractor(input_tensor)
    sample_features = features.view(features.size(0), -1)
    return sample_features

def recommend(feature_list,sample_features):
   similarity=[]
   for i in range(feature_list.shape[0]):
    similarity.append(cosine_similarity(sample_features.reshape(1,-1),feature_list[i].reshape(1,-1))[0][0])
   index_pos=sorted(list(enumerate(similarity)),reverse=True,key=lambda x:x[1])[0][0]
   return index_pos

st.title('Which bollywood celebrity are you?')
uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    # save the image in a directory
    if save_uploaded_image(uploaded_image):
        # load the image
        display_image = Image.open(uploaded_image)

        # extract the features
        features = extract_features(os.path.join('uploads', uploaded_image.name),feature_extractor,detector)
        # recommend
        index_pos = recommend(feature_list,features)
        # display
        col1,col2 = st.columns(2)

        with col1:
            st.header('Your uploaded image')
            st.image(display_image)
        with col2:
            st.header("Your Match")
            st.image(filenames[index_pos],width=300)
