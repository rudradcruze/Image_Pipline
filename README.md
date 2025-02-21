# Tomato Image Processing Pipeline

This project implements an advanced image processing pipeline for tomato images using Python. The pipeline includes preprocessing, augmentation, and feature extraction steps, leveraging libraries such as OpenCV, PyTorch, and Albumentations.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/rudradcruze/tomato-image-pipeline.git
    cd tomato-image-pipeline
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your dataset:
    - Place your images in the `./tomato_dataset` directory.
    - Ensure you have a CSV file named `_classes.csv` with columns `filename` and `label`.

2. Run the main script:
    ```sh
    python Tomato_Image_Pipline_v4.py
    ```

## Configuration

Configuration settings are defined in the `Config` class within the `Tomato_Image_Pipline_v4.py` file. Key settings include:

- `DATASET_PATH`: Path to the dataset directory.
- `CSV_FILE`: Name of the CSV file containing image filenames and labels.
- `IMAGE_SIZE`: Target size for image resizing.
- `BATCH_SIZE`: Batch size for data loading.
- `NUM_WORKERS`: Number of workers for data loading.
- `DEVICE`: Device to use for computation (CPU, CUDA, or MPS).
- `BASE_MODEL`: Base model for feature extraction (e.g., `resnet50`).
- `FEATURE_DIMENSIONS`: Number of dimensions for extracted features.
- `AUGMENTATIONS`: Dictionary specifying augmentation techniques to apply.
- `PREPROCESSING`: Dictionary specifying preprocessing techniques to apply.

## Features

- **Preprocessing**: Histogram equalization, CLAHE, denoising, morphological operations, edge detection, color space conversion, and resizing with padding.
- **Augmentation**: Horizontal flip, vertical flip, rotation, color jitter, normalization, and advanced processing filters.
- **Feature Extraction**: Using a ResNet50 model pre-trained on ImageNet.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.