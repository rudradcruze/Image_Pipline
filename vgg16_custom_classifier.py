# ================== Imports ==================
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import VGG16_Weights
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import logging

# ================== Logging Setup ==================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ================== Configuration ==================
class Config:
    TEST_FILE = "../test"
    DATASET_PATH = "../tomato_dataset"
    CSV_FILE = "_classes.csv"
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    MODEL_SAVE_PATH = "vgg16_custom_model.pth"

    @staticmethod
    def get_num_classes(csv_file):
        annotations = pd.read_csv(csv_file)
        return len(annotations['label'].unique())

    NUM_CLASSES = get_num_classes(os.path.join(DATASET_PATH, CSV_FILE))

# ================== Custom VGG16 Model ==================
class CustomVGG16(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(CustomVGG16, self).__init__()
        # Load pre-trained VGG16 model
        self.base_model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # Freeze all layers in the base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Modify the classifier head
        self.base_model.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


# ================== Dataset Class ==================
class TomatoDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index]['filename'])
        image = Image.open(img_path).convert("RGB")
        label = self.annotations.iloc[index]['label']
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


# ================== Data Preprocessing Pipeline ==================
def get_preprocessing_pipeline():
    return transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# ================== Training Function ==================
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    logger.info(f"Training Loss: {avg_loss:.4f}")


# ================== Evaluation Function ==================
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    unique_classes = set(all_labels)
    target_names = [f"Class {cls}" for cls in sorted(unique_classes)]

    report = classification_report(
        all_labels,
        all_preds,
        target_names=target_names
    )

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Classification Report:\n{report}")


# ================== Real-Life Test Function ==================
def real_life_test(model, image_path, transform, device):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    if transform:
        image = transform(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)

    logger.info(f"Predicted Class: {pred.item()}")


# ================== Main Function ==================
def main():
    # Setup paths
    csv_file = os.path.join(Config.DATASET_PATH, Config.CSV_FILE)

    # Create preprocessing pipeline
    preprocessing = get_preprocessing_pipeline()

    # Initialize datasets
    train_dataset = TomatoDataset(csv_file=csv_file, root_dir=Config.DATASET_PATH, transform=preprocessing)
    test_dataset = TomatoDataset(csv_file=csv_file, root_dir=Config.DATASET_PATH, transform=preprocessing)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    # Initialize model, loss, and optimizer
    model = CustomVGG16(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Train the model
    logger.info("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        train_model(model, train_loader, criterion, optimizer, Config.DEVICE)

    # Save the trained model
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    logger.info(f"Model saved to '{Config.MODEL_SAVE_PATH}'.")

    # Evaluate the model
    logger.info("Evaluating model on test set...")
    evaluate_model(model, test_loader, Config.DEVICE)

    # Real-life test
    logger.info("Performing real-life test...")
    test_image_path = os.path.join(Config.TEST_FILE, "test_image.jpg")  # Replace with a real test image path
    real_life_test(model, test_image_path, preprocessing, Config.DEVICE)


# Run the main function
if __name__ == "__main__":
    main()