# 📄 Brain Tumor MRI Image Classification

This project develops a deep learning-based solution for classifying brain MRI images into multiple categories according to tumor type. It involves building a custom Convolutional Neural Network (CNN) from scratch and enhancing performance through transfer learning using pre-trained models. The project culminates in a user-friendly Streamlit web application that enables real-time tumor type predictions from uploaded MRI images.

## 📌 Problem Statement

The goal is to provide an AI-powered tool to assist in the rapid and accurate classification of brain tumors from MRI scans, potentially aiding radiologists, improving patient triage, and supporting research.

## ✨ Key Features

* **Multi-Class Classification:** Distinguishes between Glioma, Meningioma, Pituitary tumors, and No Tumor.
* **Custom CNN:** Implementation of a convolutional neural network from scratch.
* **Transfer Learning:** Utilization of powerful pre-trained models (ResNet50, MobileNetV2, EfficientNetB0) for enhanced performance.
* **Data Augmentation:** Techniques applied to improve model generalization and prevent overfitting.
* **Comprehensive Evaluation:** Models are evaluated using accuracy, precision, recall, F1-score, and confusion matrices.
* **Interactive Web Application:** A Streamlit interface for easy image upload and real-time predictions.

## 📊 Dataset

* **Source:** https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset
* **License:** CC BY 4.0
* **Description:** Contains 2443 MRI images categorized into four classes: `glioma`, `meningioma`, `no_tumor`, and `pituitary`.
* **Splits:**
    * **Training Set:** 1695 images
    * **Validation Set:** 502 images
    * **Test Set:** 246 images

## 🛠 Technical Stack

* **Deep Learning Framework:** PyTorch
* **Data Manipulation:** NumPy, Pandas
* **Image Processing:** PIL (Pillow), Torchvision Transforms
* **Model Evaluation:** Scikit-learn, Matplotlib, Seaborn
* **Web Application:** Streamlit

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1. Project Setup

1.  **Clone the repository (if applicable) or create the project directory:**
    ```bash
    mkdir brain_tumor_classification
    cd brain_tumor_classification
    ```
2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    # On Windows: .\venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    ```
3.  **Install required Python packages:**
    ```bash
    # For PyTorch with CUDA 12.1 (adjust as needed for your system/GPU):
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    # For CPU only:
    # pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

    pip install numpy pandas matplotlib seaborn scikit-learn streamlit Pillow
    ```
    *(Optional: Create a `requirements.txt` file from your environment: `pip freeze > requirements.txt`)*

### 2. Dataset Acquisition

1.  Download the "Labeled MRI Brain Tumor Dataset > Version 1" from https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset.
2.  Extract the downloaded dataset into a `data/` folder in the root of your `brain_tumor_classification` project directory. Ensure the structure matches the one described in the "Project Structure" section (i.e., `data/train/glioma/`, `data/valid/meningioma/`, etc.).

### 3. Data Understanding & Preparation

1.  Navigate to the `notebooks/` directory:
    ```bash
    cd notebooks
    ```
2.  Open and run the `1_dataset_understanding.ipynb` notebook. This notebook will:
    * Verify the dataset structure and image counts.
    * Explore image properties (e.g., resolution).
    * Visualize sample images from each class.
    * *(If you encounter "file not found" errors, ensure `data_root = '../data'` is correct for the notebook's location.)*
3.  Continue in the same notebook or create `2_data_preparation.ipynb`. Implement the data preprocessing and augmentation steps using `torchvision.transforms` and `ImageFolder` to create `DataLoader` instances for training, validation, and testing.

    ```python
    # Example snippet for data preparation
    import torch
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader
    import os

    data_dir = '../data'
    image_size = (224, 224) # Standard input size for most models

    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    val_test_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_test_transforms)

    class_names = train_dataset.classes
    batch_size = 32
    num_workers = os.cpu_count() if os.name != 'nt' else 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    ```

### 4. Model Building & Training

1.  Create or open `3_model_training.ipynb` in the `notebooks/` directory.
2.  Define the `CustomCNN` class and the `create_transfer_learning_model` function as provided in Phase 3.
3.  Define the `train_model` function, which includes the training loop, early stopping, and model checkpointing.
4.  Execute the training calls for:
    * **Custom CNN:**
        ```python
        # Ensure 'models' directory exists in project root
        os.makedirs('../models', exist_ok=True)
        custom_model = CustomCNN(num_classes=len(class_names))
        trained_custom_model, custom_history = train_model(
            custom_model, train_loader, val_loader, num_epochs=30, learning_rate=0.001, model_name="custom_cnn"
        )
        ```
    * **Transfer Learning Models (ResNet50, MobileNetV2, EfficientNetB0):**
        * Train with frozen features first, then optionally fine-tune by unfreezing layers with a smaller learning rate.
        ```python
        # Example for ResNet50
        resnet_model = create_transfer_learning_model('resnet50', num_classes=len(class_names), freeze_features=True)
        trained_resnet_model, resnet_history = train_model(
            resnet_model, train_loader, val_loader, num_epochs=20, learning_rate=0.001, model_name="resnet50_frozen"
        )
        # Fine-tuning example
        resnet_model_finetune = create_transfer_learning_model('resnet50', num_classes=len(class_names), freeze_features=False)
        resnet_model_finetune.load_state_dict(torch.load('../models/resnet50_frozen_best.pth'))
        trained_resnet_model_finetune, resnet_finetune_history = train_model(
            resnet_model_finetune, train_loader, val_loader, num_epochs=10, learning_rate=0.0001, model_name="resnet50_finetuned"
        )
        # Repeat similar steps for MobileNetV2 and EfficientNetB0
        ```

### 5. Model Evaluation & Comparison

1.  Create or open `4_model_evaluation.ipynb` in the `notebooks/` directory.
2.  Define the `evaluate_model` function and `plot_training_history` function.
3.  Execute the evaluation for each trained model, loading their best saved weights.
    ```python
    # Example for EfficientNetB0
    efficientnet_model_eval = create_transfer_learning_model('efficientnet_b0', num_classes=len(class_names), freeze_features=False)
    efficientnet_labels, efficientnet_preds = evaluate_model(
        efficientnet_model_eval, test_loader, class_names, '../models/efficientnet_b0_finetuned_best.pth', "EfficientNetB0 Fine-tuned"
    )
    plot_training_history(efficientnet_history, "EfficientNetB0 Fine-tuned") # Assuming you saved history
    ```
4.  **Analyze the results:** Compare the classification reports, confusion matrices, and training history plots to determine the best-performing model.

## 📈 Model Performance Summary

*(Based on the results you provided)*

| Model                        | Overall Accuracy | Key Observations                                                                                                                                                                                                                                    |
| :--------------------------- | :--------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Custom CNN** | **0.83** | Achieved reasonable performance, but showed lower recall for `meningioma` (0.59) and lower precision for `no_tumor` (0.73), indicating room for improvement in distinguishing these classes.                                                              |
| **ResNet50 Fine-tuned** | **0.97** | Demonstrated excellent performance across all classes, with very high precision, recall, and F1-scores. `glioma` and `pituitary` classifications were near-perfect.                                                                                 |
| **MobileNetV2 Fine-tuned** | **0.95** | Performed very well, providing a robust solution, though slightly behind ResNet50 and EfficientNetB0 in overall metrics.                                                                                                                            |
| **EfficientNetB0 Fine-tuned**| **0.98** | **Outstanding performance**, achieving the highest overall accuracy. Consistently high precision, recall, and F1-scores across all tumor types, making it the most accurate and reliable classifier for this dataset.                               |

**Conclusion:** The **EfficientNetB0 Fine-tuned model** is the top performer, achieving **98% accuracy** and strong, balanced metrics across all classes. This model is selected for deployment.

## 🌐 Streamlit Application Deployment

1.  Navigate to the project root directory:
    ```bash
    cd brain_tumor_classification
    ```
2.  Create the `streamlit_app/` directory:
    ```bash
    mkdir streamlit_app
    cd streamlit_app
    ```
3.  Create an `app.py` file inside `streamlit_app/` and paste the provided Streamlit code (from Phase 5). Ensure the `MODEL_PATH` and `MODEL_NAME` variables correctly point to your chosen best model (`efficientnet_b0_finetuned_best.pth`).
4.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
    This will open the web application in your browser, allowing you to upload MRI images for classification.

## 📦 Project Deliverables

* Trained models: `custom_cnn_best.pth`, `resnet50_finetuned_best.pth`, `mobilenet_v2_finetuned_best.pth`, `efficientnet_b0_finetuned_best.pth`.
* Streamlit application for tumor classification.
* Jupyter notebooks/Python scripts for data understanding, preparation, training, and evaluation.
* Model comparison results (summarized in this README).
* Public GitHub repository with all code and documentation.