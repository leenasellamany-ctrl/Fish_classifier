import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Image Classification", layout="centered")

st.title("🧠 Image Classification App")
st.write("Upload an image and select a model to get predictions.")

# ------------------------------
# DEVICE (GPU / CPU)
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

NUM_CLASSES = len(CLASS_NAMES)

# ------------------------------
# IMAGE TRANSFORM
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ------------------------------
# LOAD MODELS (CACHED)
# ------------------------------
@st.cache_resource
def load_models():
    cnn = torch.load("models/REAL CNN TRAINING 2ND DATASET", map_location=device)
    transformer = torch.load("models/TRANSFORMER TRAINING 2ND DATASET", map_location=device)
    hybrid = torch.load("models/HYBRID TRAINING 2ND DATASET", map_location=device)

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

if model_choice == "CNN":
    model = cnn_model
elif model_choice == "Transformer":
    model = transformer_model
else:
    model = hybrid_model

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
