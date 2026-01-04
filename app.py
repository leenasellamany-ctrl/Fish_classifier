import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from transformers import SwinModel, SwinConfig
from PIL import Image
import os
import requests

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Fish Image Classification", layout="centered")

st.title("🐟 Fish Image Classification App")
st.write("Upload an image and select a model to get predictions.")

# ------------------------------
# DEVICE
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.info(f"Using device: **{device}**")

# ------------------------------
# CLASS NAMES (MUST MATCH TRAINING)
# ------------------------------
CLASS_NAMES = [
    "class_0", "class_1", "class_2", "class_3", "class_4",
    "class_5", "class_6", "class_7", "class_8"
]

NUM_CLASSES = len(CLASS_NAMES)

# ------------------------------
# IMAGE TRANSFORM
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]),
])

# ------------------------------
# GOOGLE DRIVE LINKS
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
        with st.spinner(f"Downloading {os.path.basename(save_path)}..."):
            response = requests.get(url, stream=True)
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

# ------------------------------
# MODEL DEFINITIONS
# ------------------------------
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class TransformerModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        config = SwinConfig()
        self.swin = SwinModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x):
        features = self.swin(pixel_values=x).last_hidden_state.mean(dim=1)
        return self.classifier(features)


class HybridModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Identity()

        config = SwinConfig()
        self.swin = SwinModel(config)

        self.classifier = nn.Linear(2048 + config.hidden_size, num_classes)

    def forward(self, x):
        res_feat = self.resnet(x)
        swin_feat = self.swin(pixel_values=x).last_hidden_state.mean(dim=1)
        fused = torch.cat((res_feat, swin_feat), dim=1)
        return self.classifier(fused)

# ------------------------------
# LOAD MODELS (SAFE, CACHED)
# ------------------------------
@st.cache_resource
def load_models():
    download_model(CNN_URL, CNN_PATH)
    download_model(TRANSFORMER_URL, TRANSFORMER_PATH)
    download_model(HYBRID_URL, HYBRID_PATH)

    cnn = CNNModel(NUM_CLASSES)
    cnn.load_state_dict(torch.load(CNN_PATH, map_location=device))
    cnn.to(device).eval()

    transformer = TransformerModel(NUM_CLASSES)
    transformer.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=device))
    transformer.to(device).eval()

    hybrid = HybridModel(NUM_CLASSES)
    hybrid.load_state_dict(torch.load(HYBRID_PATH, map_location=device))
    hybrid.to(device).eval()

    return cnn, transformer, hybrid

cnn_model, transformer_model, hybrid_model = load_models()

# ------------------------------
# MODEL SELECT
# ------------------------------
model_choice = st.selectbox("Select model", ["CNN", "Transformer", "Hybrid"])

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
