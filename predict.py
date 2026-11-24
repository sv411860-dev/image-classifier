import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import CatDogCNN, CatDogTransfer

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')

def load_trained_model(model_path="best_model.pth"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at '{model_path}'.")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint.get('classes', ['cats', 'dogs'])
    state_dict = checkpoint['model_state_dict']

    # Auto-detect whether checkpoint is transfer learning (MobileNet) or custom scratch CNN
    is_transfer = checkpoint.get('model_type') == 'transfer' or 'model.classifier.3.weight' in state_dict

    if is_transfer:
        model = CatDogTransfer(num_classes=len(classes), pretrained=False).to(device)
        img_size = (224, 224)
    else:
        model = CatDogCNN(num_classes=len(classes)).to(device)
        img_size = (150, 150)

    model.load_state_dict(state_dict)
    model.eval()
    return model, device, img_size

def predict_single_image(model, device, image_path, transform):
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        return None, None, f"Could not load image: {e}"

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    cat_prob = probs[0].item() * 100
    dog_prob = probs[1].item() * 100

    predicted_class = "Cat" if cat_prob > dog_prob else "Dog"
    confidence = max(cat_prob, dog_prob)

    return predicted_class, confidence, {"Cat": cat_prob, "Dog": dog_prob}

def run_predictions(input_path, model_path="best_model.pth"):
    if not os.path.exists(input_path):
        print(f"❌ Error: Path '{input_path}' does not exist.")
        return

    model, device, img_size = load_trained_model(model_path)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if os.path.isdir(input_path):
        image_files = [
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if f.lower().endswith(VALID_EXTENSIONS) and not f.startswith('.')
        ]
        if not image_files:
            print(f"⚠️ No image files ({', '.join(VALID_EXTENSIONS)}) found in directory '{input_path}'.")
            return

        print(f"\n📁 Scanning directory '{input_path}' ({len(image_files)} image{'s' if len(image_files) > 1 else ''} found):\n")
        print(f"{'Image':<35} | {'Prediction':<10} | {'Confidence':<12} | {'Cat %':<8} | {'Dog %':<8}")
        print("-" * 80)

        for img_path in sorted(image_files):
            filename = os.path.basename(img_path)
            pred, conf, details = predict_single_image(model, device, img_path, transform)
            if pred is None:
                print(f"{filename:<35} | ERROR: {details}")
            else:
                emoji = "🐱" if pred == "Cat" else "🐶"
                print(f"{filename:<35} | {emoji} {pred:<7} | {conf:>6.2f}%     | {details['Cat']:>6.2f}% | {details['Dog']:>6.2f}%")
        print("-" * 80)

    elif os.path.isfile(input_path):
        filename = os.path.basename(input_path)
        pred, conf, details = predict_single_image(model, device, input_path, transform)
        if pred is None:
            print(f"❌ Error: {details}")
            return
        emoji = "🐱" if pred == "Cat" else "🐶"
        print("=" * 45)
        print(f"File:       {filename}")
        print(f"Prediction: {emoji} {pred} ({conf:.2f}% confidence)")
        print(f"Probabilities:")
        print(f"  • 🐱 Cat: {details['Cat']:.2f}%")
        print(f"  • 🐶 Dog: {details['Dog']:.2f}%")
        print("=" * 45)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict whether an image or folder of images is Cat or Dog")
    parser.add_argument("--image", type=str, required=True, help="Path to an image file or a directory containing images")
    parser.add_argument("--model", type=str, default="best_model.pth", help="Path to model checkpoint")
    args = parser.parse_args()

    run_predictions(args.image, args.model)
