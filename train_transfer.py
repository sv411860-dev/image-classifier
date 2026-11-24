import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from model import CatDogTransfer

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    device = get_device()
    print(f"Using device for fine-tuning: {device}")

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cats_and_dogs_filtered")
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "validation")

    if not os.path.exists(train_dir):
        print(f"Dataset not found at {train_dir}.")
        return

    # MobileNetV3 standard input size: 224x224
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = train_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Initialize MobileNetV3 with pre-trained ImageNet weights
    print("\nLoading MobileNetV3 pre-trained on ImageNet...")
    model = CatDogTransfer(num_classes=len(class_names), pretrained=True).to(device)

    # Use differential learning rates: lower for pretrained backbone, higher for new head
    optimizer = optim.AdamW([
        {'params': model.model.features.parameters(), 'lr': 1e-4},
        {'params': model.model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()
    num_epochs = 5
    best_val_acc = 0.0

    print("Starting fine-tuning...")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += labels.size(0)

        epoch_val_loss = val_running_loss / total_val
        epoch_val_acc = correct_val / total_val

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.2f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_type': 'transfer',
                'model_state_dict': model.state_dict(),
                'val_acc': best_val_acc,
                'classes': class_names
            }, save_path)
            print(f"  --> Saved new best fine-tuned model (Val Acc: {best_val_acc*100:.2f}%)")

    elapsed_time = time.time() - start_time
    print(f"\nFine-tuning completed in {elapsed_time/60:.2f} minutes!")
    print(f"Highest Validation Accuracy: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    main()
