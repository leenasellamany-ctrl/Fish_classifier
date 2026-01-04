# app.py

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import gdown
import os

# -----------------------------
# 1. GPU detection
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {DEVICE}")

# -----------------------------
# 2. Google Drive model download
# -----------------------------
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Direct download links (your IDs)
MODEL_LINKS = {
    "CNN": "https://drive.google.com/uc?export=download&id=1Dj4jUHN-0FFYzgn9tY_iIGFCnOGxzgts",
    "TRANSFORMER": "https://drive.google.com/uc?export=download&id=171i6ns7niNvjhtt3Yit7n-3auKm5sywG",
    "HYBRID": "https://drive.google.com/uc?export=download&id=1S3vqcnt40uFiu-XMVhZxX5RP5DloeZWz",
}

# Download if not exists
for name, link in MODEL_LINKS.items():
    model_path = os.path.join(MODEL_DIR, f"{name}.pt")
    if not os.path.exists(model_path):
        st.write(f"Downloading {name} model...")
        gdown.download(link, model_path, quiet=False)

# -----------------------------
# 3. Load models
# -----------------------------
# Replace with your actual model classes
from cnn_model import CNNModel
from transformer_model import TransformerModel
from hybrid_model import HybridModel

cnn_model = CNNModel().to(DEVICE)
cnn_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "CNN.pt"), map_location=DEVICE))
cnn_model.eval()

transformer_model = TransformerModel().to(DEVICE)
transformer_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "TRANSFORMER.pt"), map_location=DEVICE))
transformer_model.eval()

hybrid_model = HybridModel().to(DEVICE)
hybrid_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "HYBRID.pt"), map_location=DEVICE))
hybrid_model.eval()

# -----------------------------
# 4. Image upload
# -----------------------------
st.title("Image Classification with 3 Models")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # -----------------------------
    # 5. Preprocessing
    # -----------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # -----------------------------
    # 6. Predictions
    # -----------------------------
    def get_topk_predictions(model, input_tensor, topk=3):
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            top_probs, top_idx = probs.topk(topk)
            top_probs = top_probs.cpu().numpy()[0]
            top_idx = top_idx.cpu().numpy()[0]
        return list(zip(top_idx, top_probs))

    st.subheader("Predictions:")

    for model_name, model in zip(["CNN", "Transformer", "Hybrid"], 
                                 [cnn_model, transformer_model, hybrid_model]):
        st.write(f"**{model_name} Model:**")
        predictions = get_topk_predictions(model, input_tensor, topk=3)
        for idx, prob in predictions:
            st.write(f"Class {idx}: {prob*100:.2f}%")

