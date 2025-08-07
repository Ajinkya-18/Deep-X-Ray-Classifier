# X-Ray Pneumonia Classifier

A deep learning-based binary image classifier for detecting **pneumonia** from **grayscale X-ray chest images** using **ResNet18**. This project includes model training, validation, inference, and plans for deployment via a web interface.

---

## 🚀 Project Overview

- **Objective**: Build a reliable binary classifier to detect pneumonia from chest X-rays.
- **Architecture**: ResNet18 (pretrained), adapted for 1-channel grayscale images.
- **Dataset**: Grayscale X-ray images organized into `NORMAL` and `PNEUMONIA` folders.
- **Augmentation**: Color jitter, rotation, flipping, and normalization.
- **Training**: Implemented with PyTorch and TensorBoard logging.
- **Inference**: Simple script to classify user-provided images.
- **Upcoming**: Streamlit/Gradio web app deployment for demo and testing.

---

## 📁 Directory Structure

xray-pneumonia-classifier/
│
├── data/                   # Contains training, validation, inference images (ignored in Git)
├── models/                 # Trained model weights (ResNet18)
├── notebooks/              # Jupyter notebooks for EDA, training, experimentation
├── reports/                # Logs, loss/accuracy plots, confusion matrix
├── src/                    # Source code scripts
│   ├── train.py            # Training pipeline
│   ├── test.py             # Evaluation script
│   ├── inference.py        # Run inference on user images
│   └── utils.py            # Model creation, transforms, training utils
│
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE


## Model Details
**Base model**: torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights)

**Modified input layer**: 3-channel grayscale

**Loss function**: Binary Cross Entropy (BCEWithLogitsLoss)

**Optimizer**: Adam

**Metrics**: Training/Validation Loss and Accuracy

**Epoch time**: ~3 hours/epoch for custom CNN (replaced by efficient ResNet18)

## 🧪 How to Use
**1. Clone the Repo**
git clone https://github.com/your-username/xray-pneumonia-classifier.git
cd xray-pneumonia-classifier

**2. Install Dependencies**
pip install -r requirements.txt

**3. Train the Model**
python src/train.py

**4. Evaluate the Model**
python src/test.py

**5. Run Inference on Your X-rays**
Place your images in:
data/infer/NORMAL/
data/infer/PNEUMONIA/
(above is the default structure when dataset is downloaded from the below given kaggle link.)

**Then run:**
python src/inference.py

### 📊 TensorBoard Logging
tensorboard --logdir=runs/


## 🌐 Deployment Plan
A lightweight web interface will be built using Streamlit or Gradio to:

Upload and classify X-ray images

Display model confidence

Provide results in real-time

## 🛡️ License
This project is open-sourced under the MIT License.


### 🙌 Acknowledgments
**Dataset from Kaggle**
link - https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images

PyTorch & Torchvision for backbone models
TensorBoard for visualizations

### 📬 Contact
Ajinkya Tamhankar
📧 ajinkya.tamhankar18@gmail.com