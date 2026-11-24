import os
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import CatDogCNN, CatDogTransfer

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered"
)

st.title("🐾 Cat vs Dog Image Classifier")

MODEL_PATH = "best_model.pth"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None, "None"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    classes = checkpoint.get("classes", ["cats", "dogs"])
    state_dict = checkpoint["model_state_dict"]

    is_transfer = checkpoint.get("model_type") == "transfer" or "model.classifier.3.weight" in state_dict

    if is_transfer:
        model = CatDogTransfer(num_classes=len(classes), pretrained=False).to(device)
        img_size = (224, 224)
        model_name = "MobileNetV3 (Fine-tuned)"
    else:
        model = CatDogCNN(num_classes=len(classes)).to(device)
        img_size = (150, 150)
        model_name = "Custom CNN (From Scratch)"

    model.load_state_dict(state_dict)
    model.eval()
    return model, device, img_size, model_name

model, device, img_size, model_name = load_model()

if model is None:
    st.warning("⚠️ Model weights (`best_model.pth`) not found. Please run the training script first!")
    st.code("python train_transfer.py", language="bash")
else:
    st.caption(f"Active Architecture: **{model_name}** | Running on: **{device.type.upper()}**")
    uploaded_file = st.file_uploader("Upload an image of a Cat or Dog", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        with st.spinner("Classifying image..."):
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)[0]
                cat_prob = float(probs[0].item())
                dog_prob = float(probs[1].item())

        is_cat = cat_prob > dog_prob
        winner = "🐱 Cat" if is_cat else "🐶 Dog"
        confidence = max(cat_prob, dog_prob) * 100

        st.success(f"### Prediction: **{winner}** ({confidence:.1f}%)")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="🐱 Cat Probability", value=f"{cat_prob * 100:.1f}%")
            st.progress(cat_prob)
        with col2:
            st.metric(label="🐶 Dog Probability", value=f"{dog_prob * 100:.1f}%")
            st.progress(dog_prob)
