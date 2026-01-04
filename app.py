import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import requests

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Image Classification", layout="centered")

st.title("🧠 Image Classification App")
st.write("Upload an image and select a model to get predictions.")

# ------------------------------
# DEVICE
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.info(f"Using device: **{device}**")

# ------------------------------
# CLASS NAMES (EDIT IF NEEDED)
# ------------------------------
CLASS_NAMES = [
    "class_0", "class_1", "class_2", "class_3", "class_4",
    "class_5", "class_6", "class_7", "class_8", "class_9",
    "class_10", "class_11", "class_12", "class_13", "class_14",
    "class_15", "class_16", "class_17", "class_18", "class_19",
    "class_20", "class_21", "class_22"
]

# ------------------------------
# IMAGE TRANSFORM
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ------------------------------
# GOOGLE DRIVE MODEL LINKS
# ------------------------------
CNN_URL = "https://drive.google.com/uc?id=1dy9a96bH64fxC74GTLv_Ab-yI_Q2Ll_b"
TRANSFORMER_URL = "https://drive.google.com/uc?id=1ZEyiOLoS0EUD_9Y6i37D3uiLr-KhZd9R"
HYBRID_URL = "https://drive.google.com/uc?id=1c0lcfqMl5Zg1HrCKi3jX90mAfoNl4bjf"

CNN_PATH = "models/cnn_model.pt"
TRANSFORMER_PATH = "models/transformer_model.pt"
HYBRID_PATH = "models/hybrid_model.pth"

# ------------------------------
# DOWNLOAD HELPER
# ------------------------------
def download_model(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not os.path.exists(save_path):
        with open(save_path, "wb") as f:
            response = requests.get(url, stream=True)
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

# ------------------------------
# LOAD MODELS (CACHED)
# ------------------------------
@st.cache_resource
def load_models():
    with st.spinner("Downloading & loading models (first run may take a while)..."):
        download_model(CNN_URL, CNN_PATH)
        download_model(TRANSFORMER_URL, TRANSFORMER_PATH)
        download_model(HYBRID_URL, HYBRID_PATH)

        cnn = torch.load(CNN_PATH, map_location=device)
        transformer = torch.load(TRANSFORMER_PATH, map_location=device)
        hybrid = torch.load(HYBRID_PATH, map_location=device)

        cnn.eval()
        transformer.eval()
        hybrid.eval()

        return cnn, transformer, hybrid

cnn_model, transformer_model, hybrid_model = load_models()

# ------------------------------
# MODEL SELECT
# ------------------------------
model_choice = st.selectbox(
    "Select model",
    ("CNN", "Transformer", "Hybrid")
)

model = {
    "CNN": cnn_model,
    "Transformer": transformer_model,
    "Hybrid": hybrid_model
}[model_choice]

# ------------------------------
# IMAGE UPLOAD
# ------------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# ------------------------------
# PREDICTION
# ------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1).squeeze()

    top_probs, top_idxs = torch.topk(probs, 3)

    st.subheader("🔮 Top-3 Predictions")
    for i in range(3):
        st.write(
            f"**{i+1}. {CLASS_NAMES[top_idxs[i]]}** — {top_probs[i].item()*100:.2f}%"
        )
