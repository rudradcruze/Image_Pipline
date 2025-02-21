import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

# ================== Configuration ==================
class Config:
    DATASET_PATH = "./tomato_dataset"
    CSV_FILE = "_classes.csv"
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    BASE_MODEL = "resnet50"
    FEATURE_DIMENSIONS = 512
    AUGMENTATIONS = {
        "horizontal_flip": True,
        "vertical_flip": False,
        "rotation": True,
        "color_jitter": True,
        "normalize": True,
        "advanced_processing": True,
        "filters": [
            "bilateral_filter",
            "histogram_equalization",
            "sharpening",
            "gamma_correction",
            "gaussian_blur",
        ],
    }
    PREPROCESSING = {
        "resize": True,
        "to_tensor": True,
    }


# ================== Logging Setup ==================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ================== Dataset Class ==================
class TomatoDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, preprocessing=None):
        self.annotations = pd.read_csv(csv_file)
        # Convert label columns to numeric, coercing errors to NaN
        for col in self.annotations.columns[1:]:
            self.annotations[col] = pd.to_numeric(self.annotations[col], errors='coerce')
        self.root_dir = root_dir
        self.transform = transform
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {str(e)}. Using a blank image.")
            image = Image.new("RGB", Config.IMAGE_SIZE, color=(0, 0, 0))  # Blank black image
        label_values = self.annotations.iloc[index, 1:].values.astype(np.float32)
        label = torch.tensor(label_values, dtype=torch.float32)

        if self.preprocessing:
            image = self.preprocessing(image)
            logger.debug(f"Image shape after preprocessing: {np.array(image).shape}")

        if self.transform:
            image = np.array(image)
            logger.debug(f"Image shape before augmentation: {image.shape}")
            transformed = self.transform(image=image)
            image = transformed["image"]
            logger.debug(f"Image shape after augmentation: {image.shape}")

        return image, label


def apply_advanced_image_processing(image, filters):
    """
    Apply advanced image processing techniques with proper error handling and type checking.
    """
    # Ensure the image is in the correct format
    if isinstance(image, torch.Tensor):
        logger.debug(f"Initial image shape (torch.Tensor): {image.shape}")
        image = image.permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
        logger.debug(f"Image shape after permute: {image.shape}")

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
        logger.debug(f"Image shape after dtype conversion: {image.shape}")

    # Log the shape and type of the image for debugging
    logger.debug(f"Image shape before channel check: {image.shape}, dtype: {image.dtype}")

    # Ensure the image has 3 channels
    if len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        logger.debug(f"Image shape after grayscale to RGB conversion: {image.shape}")
    elif len(image.shape) == 3 and image.shape[2] != 3:  # Incorrect number of channels
        logger.warning(f"Unexpected number of channels ({image.shape[2]}). Converting to RGB.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logger.debug(f"Image shape after channel correction: {image.shape}")

    # Make sure image is in BGR format for OpenCV operations
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        logger.debug(f"Image shape after RGB to BGR conversion: {image_bgr.shape}")
    else:
        logger.warning("Image does not have 3 channels. Returning original image.")
        return image

    try:
        if "bilateral_filter" in filters:
            image_bgr = cv2.bilateralFilter(image_bgr, d=9, sigmaColor=75, sigmaSpace=75)
        if "histogram_equalization" in filters:
            lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_eq = cv2.equalizeHist(l)
            lab = cv2.merge([l_eq, a, b])
            image_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        if "sharpening" in filters:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            image_bgr = cv2.filter2D(image_bgr, -1, kernel)
        if "gamma_correction" in filters:
            gamma = 1.5
            lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            image_bgr = cv2.LUT(image_bgr, lookup_table)
        if "gaussian_blur" in filters:
            image_bgr = cv2.GaussianBlur(image_bgr, (5, 5), 0)
        # Convert back to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        logger.debug(f"Final image shape after advanced processing: {image_rgb.shape}")
        return image_rgb
    except Exception as e:
        logger.warning(f"Error in image processing: {str(e)}. Returning original image.")
        return image

def advanced_processing_lambda(image, **kwargs):
    return apply_advanced_image_processing(image, Config.AUGMENTATIONS["filters"])


def get_augmentation_pipeline():
    transform_list = []

    # Add basic augmentations
    if Config.AUGMENTATIONS["horizontal_flip"]:
        transform_list.append(A.HorizontalFlip(p=0.5))
    if Config.AUGMENTATIONS["vertical_flip"]:
        transform_list.append(A.VerticalFlip(p=0.5))
    if Config.AUGMENTATIONS["rotation"]:
        transform_list.append(A.RandomRotate90(p=0.5))
    if Config.AUGMENTATIONS["color_jitter"]:
        transform_list.append(A.ColorJitter(p=0.5))

    # Add advanced processing
    if Config.AUGMENTATIONS["advanced_processing"]:
        transform_list.append(A.Lambda(name="AdvancedProcessing", image=advanced_processing_lambda))
        # pass

    # Always add normalization and tensor conversion at the end
    if Config.AUGMENTATIONS["normalize"]:
        transform_list.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transform_list.append(ToTensorV2())

    return A.Compose(transform_list)


def get_preprocessing_pipeline(resize_dim=(224, 224)):
    return transforms.Compose([
        transforms.Resize(resize_dim),
        # Remove ToTensor() here to keep the image as a PIL image or NumPy array
    ])


# ================== Feature Extraction Model ==================
class FeatureExtractor(nn.Module):
    def __init__(self, base_model="resnet50", feature_dim=512):
        super(FeatureExtractor, self).__init__()
        if base_model == "resnet50":
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.model.fc = nn.Linear(self.model.fc.in_features, feature_dim)
        else:
            raise ValueError(f"Unsupported base model: {base_model}")

    def forward(self, x):
        return self.model(x)

def extract_features(dataset, model, batch_size=32, device=Config.DEVICE):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())

    # Combine all batches into single arrays
    features = np.vstack(features)
    labels = np.vstack(labels)

    # Save features and labels to CSV
    feature_columns = [f"feature_{i}" for i in range(features.shape[1])]
    label_columns = [f"label_{i}" for i in range(labels.shape[1])]

    # Create a DataFrame for features and labels
    features_df = pd.DataFrame(features, columns=feature_columns)
    labels_df = pd.DataFrame(labels, columns=label_columns)

    # Combine features and labels into a single DataFrame
    combined_df = pd.concat([features_df, labels_df], axis=1)

    # Save to CSV
    csv_filename = "extracted_features_v2.csv"
    combined_df.to_csv(csv_filename, index=False)
    logger.info(f"Features and labels saved to '{csv_filename}'.")

    return features, labels

def visualize_augmentations(dataset, num_samples=5):
    fig, ax = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        image, _ = dataset[i]
        ax[i].imshow(image.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        ax[i].axis("off")
    plt.show()

def main():
    # Setup paths
    csv_file = os.path.join(Config.DATASET_PATH, Config.CSV_FILE)

    # Create preprocessing and augmentation pipelines
    preprocessing = get_preprocessing_pipeline(Config.IMAGE_SIZE)
    augmentation = get_augmentation_pipeline()

    # Initialize dataset
    dataset = TomatoDataset(
        csv_file=csv_file,
        root_dir=Config.DATASET_PATH,
        transform=augmentation,
        preprocessing=preprocessing
    )

    visualize_augmentations(dataset)

    # Initialize feature extractor
    feature_extractor = FeatureExtractor(
        base_model=Config.BASE_MODEL,
        feature_dim=Config.FEATURE_DIMENSIONS
    ).to(Config.DEVICE)

    # Extract features
    logger.info("Starting feature extraction...")
    features, labels = extract_features(dataset, feature_extractor)

    logger.info(f"Features Shape: {features.shape}")
    logger.info(f"Labels Shape: {labels.shape}")


if __name__ == "__main__":
    main()