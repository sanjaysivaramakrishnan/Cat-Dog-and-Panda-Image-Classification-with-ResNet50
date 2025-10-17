import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Animal Classifier",
    page_icon="🐾",
    layout="centered"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        font-size: 1.1rem;
        border-radius: 8px;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Device configuration
@st.cache_resource
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
@st.cache_resource
def load_model():
    device = get_device()
    
    # Load ResNet50 architecture
    model = models.resnet50(weights=None)
    
    # Modify the final layer to match your trained model BEFORE loading weights
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.6),  # Updated to match your model
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 3)  # 3 classes: cat, dog, panda
    )
    
    # Move model to device first
    model = model.to(device)
    
    # Load the saved weights
    # Replace 'model.pth' with your actual model file path
    try:
        # Use strict=False to ignore missing/unexpected keys if needed
        state_dict = torch.load('model.pth', map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please ensure 'model.pth' is in the same directory.")
        return None, device
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None, device

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.Resize((224, 254)),  # Match your test_transform
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(model, image, device):
    # Preprocess the image
    image_tensor = preprocess_image(image).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item(), probabilities[0].cpu().numpy()

# Main app
def main():
    # Header
    st.title("🐾 Animal Classifier")
    st.markdown("### Classify images of Cats, Dogs, and Pandas")
    st.markdown("---")
    
    # Class names
    class_names = ['Cat', 'Dog', 'Panda']
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.info("💡 **Instructions to use this app:**")
        st.markdown("""
        1. Save your trained model using: `torch.save(model.state_dict(), 'model.pth')`
        2. Place the `model.pth` file in the same directory as this script
        3. Run the app again: `streamlit run app.py`
        """)
        return
    
    # Device info
    with st.expander("ℹ️ System Information"):
        st.write(f"**Device:** {device}")
        st.write(f"**Classes:** {', '.join(class_names)}")
    
    # File uploader
    st.markdown("### Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image of a cat, dog, or panda"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        # Predict button
        if st.button("🔍 Classify Image"):
            with st.spinner('Analyzing image...'):
                # Make prediction
                pred_class, confidence, probabilities = predict(model, image, device)
                
                with col2:
                    st.markdown("#### Prediction Results")
                    
                    # Display prediction
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2 style='text-align: center; color: #4CAF50;'>{class_names[pred_class]}</h2>
                        <h4 style='text-align: center;'>Confidence: {confidence*100:.2f}%</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display all probabilities
                    st.markdown("#### Confidence Scores")
                    for i, (name, prob) in enumerate(zip(class_names, probabilities)):
                        st.progress(float(prob), text=f"{name}: {prob*100:.2f}%")
    
    else:
        # Sample instructions
        st.info("👆 Upload an image to get started!")
        
        # Example section
        with st.expander("📖 How to use"):
            st.markdown("""
            1. **Upload** an image using the file uploader above
            2. Click the **Classify Image** button
            3. View the **prediction results** with confidence scores
            
            **Supported formats:** JPG, JPEG, PNG
            
            **Note:** The model works best with clear images of cats, dogs, or pandas.
            """)

if __name__ == "__main__":
    main()