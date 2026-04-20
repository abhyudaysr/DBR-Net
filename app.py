import streamlit as st
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from streamlit_image_comparison import image_comparison


# 1. CONFIGURATION & IMPORTS

# Define the device globally
# This will use the GPU (cuda) if available, otherwise fall back to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # Make sure this import matches your project's file structure!
    from models.dbr_net import DBRNet
except ImportError:
    st.error("⚠️ Could not import 'DBRNet'. Please check the import path in app.py.")
    st.stop()

# Path to your best saved model weights
MODEL_PATH = "results/checkpoints/dbrnet_best.pth"


# 2. MODEL LOADING (ON GPU)

@st.cache_resource
def load_model():
    """
    Loads the DBR-Net model and moves it to the GPU.
    """
    try:
        st.sidebar.info(f"Loading model on: **{DEVICE}**")

        # 1. Initialize the architecture
        model = DBRNet()

        # 2. Load weights
        # We remove map_location='cpu' so it loads onto the GPU directly
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

        # 3. Handle different saving formats
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        # 4. Move the entire model to the GPU
        model = model.to(DEVICE)

        model.eval()  # Set to evaluation mode
        st.sidebar.success(f"✅ Model successfully loaded on {DEVICE}!")
        return model

    except FileNotFoundError:
        st.error(f"❌ Error: Could not find model file at {MODEL_PATH}.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()


# 3. INFERENCE LOGIC (ON GPU)

def enhance_image(original_image, model):
    """
    Runs the DBR-Net model on the GPU.
    """
    # 1. Preprocess
    transform = T.Compose([
        T.ToTensor()
    ])

    # Create the tensor and immediately move it to the GPU
    img_tensor = transform(original_image).unsqueeze(0).to(DEVICE)

    # 2. Predict
    with torch.no_grad():
        # The model is already on the GPU, so this calculation happens there.
        output_tensor = model(img_tensor)

    # 3. Postprocess
    # Move the result back to the CPU for converting to an image
    output_tensor = output_tensor.cpu().squeeze(0).clamp(0, 1)

    to_pil = T.ToPILImage()
    enhanced_pil = to_pil(output_tensor)

    # Resize back to original size for comparison
    enhanced_pil = enhanced_pil.resize(original_image.size)

    return np.array(enhanced_pil)


# 4. THE GUI (STREAMLIT)

st.set_page_config(page_title="DBR-Net Demo", layout="centered")

st.title("🌊 DBR-Net: Underwater Image Enhancement")
st.markdown("Upload a raw underwater image to see the **Dual-Branch Refinement Network** in action.")

# Load Model
model = load_model()

# Sidebar
st.sidebar.header("Settings")

# File Uploader
uploaded_file = st.file_uploader("Choose an underwater image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    original_image = Image.open(uploaded_file).convert('RGB')

    # Run Inference
    with st.spinner('Enhancing Image with DBR-Net on GPU...'):
        enhanced_result = enhance_image(original_image, model)

    st.success("Enhancement Complete!")

    # Comparison Slider
    st.markdown("### 🔍 Before vs. After")
    image_comparison(
        img1=original_image,
        img2=enhanced_result,
        label1="Raw Input",
        label2="DBR-Net Enhanced",
        width=700,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True,
    )