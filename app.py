import streamlit as st
import torch.nn as nn
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os

# --- Configuration ---
MODEL_PATH = 'models/efficientnet_b0_finetuned_best.pth'
MODEL_NAME = 'efficientnet_b0' # Corresponds to the model used in create_transfer_learning_model
NUM_CLASSES = 4 # Number of classes in your dataset
IMAGE_SIZE = (224, 224) # Standard input size for EfficientNetB0
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definition  ---
def create_transfer_learning_model(model_name, num_classes, pretrained=False):
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    else:
        raise ValueError(f"Model {model_name} not supported for deployment.")
    return model

# --- Load the trained model ---
@st.cache_resource # Cache the model so it's loaded only once
def load_model(model_path, model_name, num_classes):
    model = create_transfer_learning_model(model_name, num_classes, pretrained=False)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval() # Set to evaluation mode
        return model
    else:
        st.error(f"Model file not found at: {model_path}")
        return None

model = load_model(MODEL_PATH, MODEL_NAME, NUM_CLASSES)

# --- Image Preprocessing for Inference ---
inference_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.Grayscale(num_output_channels=3), # Ensure 3 channels for models like EfficientNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Same normalization as training
])

# --- Prediction Function ---
def predict_image(image):
    if model is None:
        return "Model Not Loaded", 0.0

    # Preprocess the image
    image = inference_transforms(image).unsqueeze(0).to(DEVICE) # Add batch dimension and move to device

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    return predicted_class, confidence.item()

# --- Streamlit UI ---
st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="centered")

st.title("🧠 Brain Tumor MRI Image Classifier")
st.markdown("""
Upload a Brain MRI image and let the model predict the tumor type (Glioma, Meningioma, No Tumor, Pituitary).
""")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB") # Convert to RGB to handle all image types
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make prediction
    label, confidence = predict_image(image)

    if label != "Model Not Loaded":
        st.success(f"Prediction: **{label.replace('_', ' ').title()}**")
        st.info(f"Confidence: **{confidence*100:.2f}%**")

        st.markdown("---")
        st.subheader("What do these mean?")
        st.write(f"The model predicts this image most closely matches a **'{label.replace('_', ' ').title()}'** case based on its training.")
        st.write("A higher confidence percentage indicates the model is more certain about its prediction.")

    else:
        st.error("Could not make a prediction. Please check if the model is loaded correctly.")

st.markdown("---")
st.markdown("""
*Note: This is an AI-powered classification tool for educational/demonstration purposes and should not be used for actual medical diagnosis. Always consult with a qualified medical professional for diagnosis and treatment.*
""")