# Cat, Dog, and Panda Image Classification with ResNet50

This project implements a deep learning model using PyTorch and transfer learning with ResNet50 to classify images of cats, dogs, and pandas. The model achieves high accuracy by leveraging pre-trained weights and fine-tuning for the specific classification task.

## Overview

The project uses transfer learning with a pre-trained ResNet50 model from torchvision to classify images into three categories:
- **Cat** (Class 0)
- **Dog** (Class 1) 
- **Panda** (Class 2)

Key features:
- Transfer learning with ResNet50 backbone
- Custom data loading and preprocessing
- Training with validation monitoring
- Model evaluation on test images
- GPU acceleration support
- Progress tracking with tqdm

## Dataset

This project uses the "Cat-Dog_Pandas" dataset from Kaggle. The dataset contains:
- Training images in `/Train` directory
- Validation images in `/Valid` directory  
- Test images in `/Test` directory

**Dataset Link**: [Cat-Dog_Pandas Dataset on Kaggle](https://www.kaggle.com/datasets/cat-dot-pandas-dataset)

## Requirements

### System Requirements
- Python 3.7+
- CUDA-compatible GPU (recommended) or CPU
- At least 4GB RAM
- 2GB+ storage space for dataset

### Python Dependencies
Install the required packages using:

```bash
pip install -r requirements.txt
```

## Setup Instructions

### 1. Environment Setup

#### Option A: Local Setup
```bash
# Clone or download the project
git clone <your-repo-url>
cd cat-dog-panda-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Kaggle Notebook (Recommended)
This project is optimized for Kaggle notebooks with built-in GPU support:
1. Go to [Kaggle](https://www.kaggle.com)
2. Create a new notebook
3. Add the "Cat-Dog_Pandas" dataset to your notebook
4. Enable GPU acceleration: Settings → Accelerator → GPU
5. Copy and paste the code

### 2. CUDA Check

Before running the model, verify CUDA availability:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

Expected output with GPU:
```
CUDA available: True
CUDA version: 11.8
Device: cuda
```

### 3. Dataset Preparation

#### For Kaggle Users:
1. Add the "Cat-Dog_Pandas" dataset to your notebook
2. The dataset will be available at `/kaggle/input/cat-dot-pandas-dataset/`
3. No additional setup required

#### For Local Users:
1. Download the dataset from Kaggle
2. Extract to your project directory
3. Update file paths in the code:
   ```python
   # Change from:
   train_data = ImageFolder('/kaggle/input/cat-dot-pandas-dataset/Cat-Dog_Pandas/Train', ...)
   
   # To:
   train_data = ImageFolder('./Cat-Dog_Pandas/Train', ...)
   ```

## Usage

### Training the Model

1. **Load and preprocess data**: The script automatically handles data loading with appropriate transforms
2. **Configure model**: ResNet50 with modified final layer for 3-class classification
3. **Train**: Run the training loop for 15 epochs with Adam optimizer
4. **Monitor**: Track training and validation metrics in real-time

```bash
python train_model.py
```

### Model Evaluation

The script includes functionality to test the trained model on unseen images:

```python
# Test on specific images
RandomImagePrediction("path/to/cat/image.jpg")    # Should predict: Cat
RandomImagePrediction("path/to/dog/image.jpg")    # Should predict: Dog  
RandomImagePrediction("path/to/panda/image.jpg")  # Should predict: Pandas
```

### Saving and Loading Models

The trained model is automatically saved as `resnet50_cat_dog_panda.pth`:

```python
# Save model
torch.save(model.state_dict(), "resnet50_cat_dog_panda.pth")

# Load model
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("resnet50_cat_dog_panda.pth"))
model.eval()
```

## Kaggle-Specific Instructions

### Setting up Kaggle API (for local development):

1. **Install Kaggle API**:
   ```bash
   pip install kaggle
   ```

2. **Get API credentials**:
   - Go to Kaggle → Account → API → Create New API Token
   - Download `kaggle.json` to `~/.kaggle/kaggle.json`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. **Download dataset**:
   ```bash
   kaggle datasets download -d <dataset-name>
   unzip <dataset-name>.zip
   ```

### Kaggle Notebook Benefits:
- Pre-installed deep learning libraries
- Free GPU access (30 hours/week)
- Easy dataset integration
- Built-in version control
- Community sharing capabilities

## Model Architecture

- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Input Size**: 224x224 RGB images
- **Output Classes**: 3 (Cat, Dog, Panda)
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (lr=0.001)
- **Training Epochs**: 15

## Performance

The model typically achieves:
- Training Accuracy: >95%
- Validation Accuracy: >90%
- Fast inference on GPU (~0.01s per image)

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   - Reduce batch size from 32 to 16 or 8
   - Clear GPU cache: `torch.cuda.empty_cache()`

2. **Dataset Path Errors**:
   - Verify dataset location
   - Check file permissions
   - Ensure correct directory structure

3. **Import Errors**:
   - Verify all dependencies are installed
   - Check Python version compatibility
   - Restart kernel/environment

### Memory Optimization:
```python
# Reduce batch size
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

# Clear cache periodically
if epoch % 5 == 0:
    torch.cuda.empty_cache()
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request


## Acknowledgments

- PyTorch team for the excellent deep learning framework
- torchvision for pre-trained models
- Kaggle for dataset hosting and compute resources
- ResNet paper authors for the architecture innovation