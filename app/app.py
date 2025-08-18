import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import io
from PIL import Image
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_model, val_transforms


@st.cache_resource
def get_model():
    return load_model('../models/x_ray_classifier_resnet18-layer4-fc-unfrozen_v1.pt')

model = get_model()


st.title("X-Ray Pneumonia Classifier")
st.write("Upload a chest X-Ray Image to predict Pneumonia (Positive/Normal).")

uploaded_file = st.file_uploader("Choose an X-Ray Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-Ray Image', use_container_width=True)

    # Preprocessing
    img_tensor = val_transforms(image).unsqueeze(0)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
        pred = 'Pneumonia' if prob > 0.5 else 'Normal'   # threshold can be changed to 0.74 based on the class (imbalance) weights

    st.markdown(f'### Prediction: **{pred}**')
    st.markdown(f'Confidence: **{prob*100:.2f} %**')





