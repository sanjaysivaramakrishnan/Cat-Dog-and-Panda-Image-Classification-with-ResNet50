---
title: Animal Classifier 🐾
emoji: 🐾
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
python_version: 3.9
---

# 🐾 Animal Classifier

A state-of-the-art deep learning application that classifies images as **Cat**, **Dog**, or **Panda** using ResNet50 transfer learning with a beautiful interactive Streamlit interface.

## 🎯 Features

### 📸 **Smart Image Classification**
- Upload images in JPG, JPEG, PNG, or WebP formats
- Real-time classification with confidence scores
- Interactive probability visualization with Plotly charts
- Support for batch processing

### 📊 **Comprehensive Analytics**
- **Performance Metrics**: Detailed per-class precision, recall, F1-scores
- **Confusion Matrix**: Interactive heatmap visualization
- **Training History**: Loss and accuracy plots over epochs
- **Model Details**: Architecture insights and specifications

### 🎨 **Beautiful Interface**
- Modern, responsive UI with custom CSS styling
- Mobile-friendly design
- Dark/light theme compatibility
- Intuitive navigation with sidebar controls

## 🏗️ Model Architecture

This project uses **transfer learning** with a pre-trained ResNet50 model:

- **Base Model**: ResNet50 (ImageNet pre-trained)
- **Fine-tuned Layers**: Layer4 + Custom Classifier
- **Custom Head**: 
  - Linear(2048 → 512) + ReLU + Dropout(0.7)
  - Linear(512 → 128) + ReLU + Dropout(0.3) 
  - Linear(128 → 3) [Output layer]
- **Classes**: Cat 🐱, Dog 🐕, Panda 🐼

## 📊 Performance Metrics

- **Test Accuracy**: **99.33%**
- **Cat Classification**: 99.00% F1-Score
- **Dog Classification**: 98.99% F1-Score  
- **Panda Classification**: 100.00% (Perfect!)

## 📁 Project Structure

```
unit-2/
├── 🐳 Dockerfile                           # Docker configuration for deployment
├── 📱 app.py                              # Main Streamlit application (801 lines)
├── 📓 cat-dog-pandas-classification.ipynb # Complete training notebook
├── 🧠 model.pth                           # Trained ResNet50 weights (94MB)
├── 📈 metrics.json                        # Comprehensive performance data
├── 📋 requirements.txt                    # Optimized dependencies (CPU PyTorch)
├── 🖼️ confusion_matrix.png               # Confusion matrix visualization
├── 🔧 datasplit.py                       # Dataset preparation utilities
└── 📁 Cat-Dog_Pandas-dataset/            # Complete dataset (2,700 images)
    ├── Train/ (2,100 images: 700 each class)
    ├── Valid/ (300 images: 100 each class) 
    └── Test/ (300 images: 100 each class)
```

## 🚀 Quick Start

### Local Development

1. **Clone & Navigate:**
   ```bash
   git clone <your-repo>
   cd unit-2
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Application:**
   ```bash
   streamlit run app.py
   ```

4. **Open Browser:** `http://localhost:8501`

### Docker Deployment

1. **Build Image:**
   ```bash
   docker build -t animal-classifier .
   ```

2. **Run Container:**
   ```bash
   docker run -p 7860:7860 animal-classifier
   ```

3. **Access App:** `http://localhost:7860`

## 🤗 Hugging Face Spaces Deployment

### Docker-based Deployment (Recommended)

This project is optimized for **Docker deployment** on Hugging Face Spaces:

#### **Step 1: Create Space**
- Go to [Hugging Face Spaces](https://huggingface.co/new-space)
- Choose **Docker** as SDK
- Set **app_port: 7860**
- Select visibility (Public/Private)

#### **Step 2: Upload Files**
Upload all project files via Git or web interface:

```bash
git clone https://huggingface.co/spaces/yourusername/animal-classifier
cd animal-classifier

# Copy project files
cp unit-2/* .

# For large model files (>10MB), use Git LFS
git lfs install
git lfs track "*.pth"
git add .gitattributes

# Deploy
git add .
git commit -m "Deploy animal classifier with Docker"
git push
```

#### **Step 3: Configuration**
The project includes optimal configuration:
- **🐳 Dockerfile**: Multi-stage build with security best practices
- **📦 requirements.txt**: CPU-optimized PyTorch for faster deployment
- **📝 README.md**: Proper Hugging Face Space headers

### **Deployment Benefits:**
- ⚡ **Fast Build**: CPU-only PyTorch (~3-5 min build time)
- 💾 **Efficient**: ~800MB final image (vs 2GB+ with full PyTorch)
- 🔒 **Secure**: Non-root user, proper health checks
- 💰 **Cost-effective**: Runs perfectly on free CPU tier

## 🛠️ Technical Details

### **Dependencies (Optimized for Docker)**
```txt
streamlit==1.31.0          # Web framework
torch==2.1.0+cpu          # PyTorch (CPU-optimized)
torchvision==0.16.0+cpu   # Computer vision utilities
numpy==1.24.3             # Numerical computing
pandas==2.1.3             # Data manipulation
pillow==10.1.0             # Image processing
plotly==5.18.0             # Interactive visualizations
```

### **Training Configuration**
- **Optimizer**: AdamW (lr=5e-6, weight_decay=5e-4)
- **Loss Function**: CrossEntropyLoss with Label Smoothing (0.1)
- **Batch Size**: 32
- **Max Epochs**: 20 (Early Stopping: patience=5)
- **Data Augmentation**: Random crop, flip, rotation, color jitter

### **Model Specifications**
- **Total Parameters**: ~24M
- **Trainable Parameters**: ~16M
- **Input Size**: 224×224 RGB
- **Normalization**: ImageNet statistics
- **Device Support**: CPU/GPU (auto-detection)

## 💡 Usage Examples

### **Basic Classification**
1. Launch the application
2. Navigate to **🔮 Prediction** tab
3. Upload an image (JPG/PNG/WebP)
4. Click **🔍 Classify Image**
5. View results with confidence scores

### **Performance Analysis**
1. Go to **📊 Model Metrics** tab
2. Explore **Overview** for key metrics
3. Check **Confusion Matrix** for detailed analysis
4. Review **Training History** for learning curves

### **API Integration** (Advanced)
The model can be loaded programmatically:
```python
import torch
from torchvision import models

# Load model
model = models.resnet50(weights=None)
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()
```

## 📈 Dataset Information

**Cat-Dog-Pandas Dataset** (Custom curated):
- **Total Images**: 2,700 high-quality images
- **Training Set**: 2,100 images (70%)
- **Validation Set**: 300 images (15%)
- **Test Set**: 300 images (15%)
- **Classes**: Balanced distribution (900 images per class)
- **Resolution**: Variable (resized to 224×224 during training)

## 🔧 Development & Customization

### **Local Development Setup**
```bash
# Clone repository
git clone <your-repo-url>
cd unit-2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
streamlit run app.py
```

### **Model Retraining**
Use the included Jupyter notebook:
```bash
jupyter notebook cat-dog-pandas-classification.ipynb
```

### **Adding New Classes**
1. Update dataset structure in `Cat-Dog_Pandas-dataset/`
2. Modify `class_names` in `app.py`
3. Retrain model with updated data
4. Update `model.pth` with new weights

## 🚀 Production Deployment Options

### **1. Hugging Face Spaces (Recommended)**
- ✅ Free hosting
- ✅ Automatic HTTPS
- ✅ Global CDN
- ✅ Built-in analytics
- ✅ Easy sharing

### **2. Docker Self-hosted**
```bash
docker build -t animal-classifier .
docker run -d -p 7860:7860 --name classifier animal-classifier
```

### **3. Cloud Platforms**
- **AWS**: ECS/Fargate deployment
- **Google Cloud**: Cloud Run
- **Azure**: Container Instances
- **Railway**: Direct Docker deployment

## 📚 Project Highlights

- 🎯 **State-of-the-art Accuracy**: 99.33% test accuracy
- 🏗️ **Production Ready**: Docker containerized, optimized for deployment
- 🎨 **Beautiful UI**: Modern Streamlit interface with custom styling
- 📊 **Comprehensive Analytics**: Full performance metrics and visualizations
- 🔬 **Reproducible**: Complete training pipeline in Jupyter notebook
- ⚡ **Fast Inference**: CPU-optimized for real-time predictions
- 📱 **Mobile Friendly**: Responsive design for all devices
- 🔒 **Secure**: Docker best practices with non-root user

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## 📄 License

This project is open source and available under the Apache 2.0 License.


---

**Built with ❤️ using PyTorch, Streamlit, and Docker by Sanjay Sivaramakrishnan** 

*Ready for production deployment on Hugging Face Spaces! 🚀*
