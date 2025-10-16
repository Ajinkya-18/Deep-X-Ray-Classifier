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
├── app/                    # contains the app.py file for running streamlit based interface.
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
**Base model**: torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

**Modified input layer**: 3-channel grayscale

**Loss function**: Binary Cross Entropy (BCEWithLogitsLoss)

**Optimizer**: Adam

**Metrics**: Training/Validation Loss and Accuracy

**Epoch time**: ~3 hours/epoch for custom CNN (replaced by efficient ResNet18 --> ~5 mins/epoch)

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

---

## Implementation details (in-depth)

This section summarizes the concrete implementation in `src/` and the exact model artifacts produced by training.

1) Project scripts

- `src/train.py` — training entry point. It builds DataLoaders from `data/chest_xray/train` and `data/chest_xray/test` using the transforms defined in `src/utils.py`, initializes a ResNet18 backbone (pretrained weights), replaces the `fc` head with a small MLP, freezes backbone weights by default, then trains for up to 50 epochs with BCEWithLogitsLoss, Adam optimizer and ReduceLROnPlateau scheduling. The best model (by validation loss) is saved to `models/x_ray_classifier_resnet18-layer4-fc-unfrozen_v1.pt`.

- `src/test.py` — evaluation script. Loads `../models/x_ray_classifier_resnet18-layer4-fc-unfrozen_v1.pt` via `utils.load_model()` and runs the `test_model()` routine on `data/test` using the same validation transforms.

- `src/inference.py` — inference script. Loads `../models/x_ray_classifier_resnet18-layer4-fc-unfrozen_v1.pt` (or another model file you provide) and runs predictions on images placed under `data/infer`. Outputs predicted class and confidence for each image.

- `src/utils.py` — shared utilities: transforms, model initialization (`initialize_model()`), train/test loops (`train_model()`, `test_model()`), model saving/loading helpers and an `EarlyStopping` helper.


### Transforms and preprocessing

- Training transforms (`train_transforms`) apply: convert to 3-channel grayscale, resize to 256x256, random horizontal flip, small rotation, Gaussian blur, conversion to tensor/dtype, and normalization (ImageNet mean/std). Validation transforms (`val_transforms`) perform grayscale-to-3, resize, dtype conversion and normalization.

### Model architecture

- Base backbone: `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`

- Modified final head: sequential MLP replacing `model.fc`:
  - Linear(in_features, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, num_classes)

- Backbone is frozen by default; training focuses on the custom head. The code contains commented lines showing how to unfreeze layer-4 if you want to fine-tune.

### Training details

- Loss: `BCEWithLogitsLoss` (binary classification)
- Optimizer: `Adam` with ReduceLROnPlateau scheduler
- Early stopping: implemented with patience and min_delta in `utils.EarlyStopping`.
- Logging: TensorBoard writer writes to `../reports/exp2_resnet18-layer4-fc-unfrozen` by default.

### Artifacts and filenames

- Trained model saved by default to: `models/x_ray_classifier_resnet18-layer4-fc-unfrozen_v1.pt` (train script overwrites when validation loss improves).
- Example model filenames you may see in `models/` (repository may contain earlier versions):
  - `x_ray_classifier_resnet18-layer4-fc-unfrozen_v1.pt`
  - (other variants may exist depending on experiments)

### Quick commands (copyable)

```powershell
# create env & install
pip install -r requirements.txt

# train
python src/train.py

# test/evaluate
python src/test.py

# inference (places images under data/infer/)
python src/inference.py

# start tensorboard (from repo root)
tensorboard --logdir=reports/
```

### Notes and tips

- The code uses CPU by default but will automatically use CUDA if available for inference (see `src/inference.py`). Training code in `src/train.py` currently sets device to `cpu` — you can modify the `device` variable and DataLoader options for GPU acceleration.
- If you want to fine-tune more layers, uncomment the appropriate lines in `src/utils.initialize_model` that unfreeze `layer4` parameters.
- Keep an eye on the saved checkpoints under `models/` and the TensorBoard logs in `reports/` to compare experiments.

---

If you want, I can:

- update `src/train.py` to accept CLI args (data paths, epochs, batch size, device) using `argparse` so the script is easier to run;
- add a small Streamlit demo in `app/` (already present) wired to load the best model and predict uploaded images.

### 📊 TensorBoard Logging

```powershell
tensorboard --logdir=runs/
```

## 🌐 Deployment Plan
A lightweight web interface built using Streamlit to:

Upload and classify X-ray images

Display model confidence

Provide results in real-time

### Streamlit App link
- [X-Ray Classifier](https://deep-x-ray-classifier-bwpdkh2gmjw8aupt5muygp.streamlit.app/)


## 🛡️ License
This project is open-sourced under the MIT License.


### 🙌 Acknowledgments

#### Dataset (Kaggle)

- [Labeled Chest Xray Images (Kaggle)](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images)

- PyTorch & Torchvision for backbone models
- TensorBoard for visualizations

### 📬 Contact

Ajinkya Tamhankar
<ajinkya.tamhankar18@gmail.com>


