import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import json
import os

# Page configuration
st.set_page_config(
    page_title="Pet Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-size: 1.1rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Model loading function
@st.cache_resource
def load_model(model_path):
    """
    Load your PyTorch model here.
    Modify this function according to your model architecture.
    """
    try:
        # Example for a ResNet-based model
        model = models.resnet50(pretrained=False)
        num_classes = 3  # cat, dog, panda
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Load your trained weights
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Image preprocessing
def preprocess_image(image):
    """
    Preprocess the uploaded image for model inference.
    Adjust according to your model's requirements.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(model, image_tensor, class_names=['Cat', 'Dog', 'Panda']):
    """
    Make prediction on the input image.
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    pred_class = class_names[predicted.item()]
    conf_score = confidence.item() * 100
    all_probs = {class_names[i]: probabilities[0][i].item() * 100 
                 for i in range(len(class_names))}
    
    return pred_class, conf_score, all_probs

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2138/2138440.png", width=100)
    st.title("🐾 Pet Classifier")
    st.markdown("---")
    
    # Model configuration
    st.subheader("Model Settings")
    model_path = st.text_input("./model.pth", "model.pth", 
                               help="Path to your trained PyTorch model")
    
    confidence_threshold = st.slider("Confidence Threshold (%)", 
                                    min_value=0, max_value=100, value=50)
    
    st.markdown("---")
    st.markdown("### About")
    st.info("This application uses a deep learning model to classify images of cats, dogs, and pandas.")
    
    if st.button("Clear History"):
        st.session_state.prediction_history = []
        st.success("History cleared!")

# Main content
st.title("🐾 Cat, Dog & Panda Classifier")
st.markdown("Upload an image to classify whether it's a **Cat**, **Dog**, or **Panda**!")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📸 Image Classification", "📊 Model Performance", "📈 Prediction History"])

# Tab 1: Image Classification
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", 
                                        type=['png', 'jpg', 'jpeg'],
                                        help="Upload a clear image of a cat, dog, or panda")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
            if st.button("🔍 Classify Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Load model
                    model = load_model(model_path)
                    
                    if model is not None:
                        # Preprocess and predict
                        image_tensor = preprocess_image(image)
                        pred_class, conf_score, all_probs = predict(model, image_tensor)
                        
                        # Store in history
                        st.session_state.prediction_history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'prediction': pred_class,
                            'confidence': conf_score,
                            'probabilities': all_probs
                        })
                        
                        # Display results in col2
                        with col2:
                            st.subheader("Prediction Results")
                            
                            # Main prediction
                            if conf_score >= confidence_threshold:
                                st.markdown(f"""
                                <div class="prediction-box">
                                    <h1>🎯 {pred_class}</h1>
                                    <h2>{conf_score:.2f}% Confidence</h2>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.warning(f"⚠️ Low confidence prediction: {pred_class} ({conf_score:.2f}%)")
                            
                            # Probability bar chart
                            st.markdown("### Class Probabilities")
                            prob_df = pd.DataFrame(list(all_probs.items()), 
                                                  columns=['Class', 'Probability'])
                            
                            fig = px.bar(prob_df, x='Class', y='Probability',
                                       color='Probability',
                                       color_continuous_scale='viridis',
                                       text='Probability')
                            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                            fig.update_layout(showlegend=False, height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Detailed probabilities
                            with st.expander("📋 Detailed Probabilities"):
                                for cls, prob in all_probs.items():
                                    st.metric(cls, f"{prob:.2f}%")
    
    with col2:
        if uploaded_file is None:
            st.info("👈 Upload an image to get started!")
            st.markdown("### Sample Images")
            st.markdown("""
            For best results:
            - Use clear, well-lit images
            - Ensure the animal is the main subject
            - Avoid blurry or low-quality images
            """)

# Tab 2: Model Performance
with tab2:
    st.header("📊 Model Performance Metrics")
    
    # Note: Replace these with your actual model metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Overall Accuracy", "94.5%", "2.3%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Precision", "93.2%", "1.8%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Recall", "92.8%", "1.5%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("F1-Score", "93.0%", "1.6%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Class-wise performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Class-wise Accuracy")
        class_accuracy = pd.DataFrame({
            'Class': ['Cat', 'Dog', 'Panda'],
            'Accuracy': [95.2, 93.8, 94.5],
            'Samples': [1200, 1350, 980]
        })
        
        fig = px.bar(class_accuracy, x='Class', y='Accuracy',
                    color='Accuracy', text='Accuracy',
                    color_continuous_scale='blues')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Confusion Matrix")
        # Sample confusion matrix (replace with actual data)
        confusion_matrix = np.array([[1142, 38, 20],
                                     [45, 1267, 38],
                                     [25, 29, 926]])
        
        fig = px.imshow(confusion_matrix,
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Cat', 'Dog', 'Panda'],
                       y=['Cat', 'Dog', 'Panda'],
                       color_continuous_scale='RdYlGn',
                       text_auto=True)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Training history
    st.subheader("Training History")
    
    # Sample training data (replace with actual data)
    epochs = list(range(1, 51))
    train_acc = [60 + i * 0.7 + np.random.randn() * 2 for i in range(50)]
    val_acc = [58 + i * 0.7 + np.random.randn() * 2.5 for i in range(50)]
    train_loss = [2.5 - i * 0.04 + np.random.randn() * 0.05 for i in range(50)]
    val_loss = [2.6 - i * 0.04 + np.random.randn() * 0.08 for i in range(50)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_acc, name='Train Accuracy', 
                                line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=epochs, y=val_acc, name='Val Accuracy',
                                line=dict(color='red', width=2)))
        fig.update_layout(title='Model Accuracy', xaxis_title='Epoch',
                         yaxis_title='Accuracy (%)', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_loss, name='Train Loss',
                                line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=epochs, y=val_loss, name='Val Loss',
                                line=dict(color='red', width=2)))
        fig.update_layout(title='Model Loss', xaxis_title='Epoch',
                         yaxis_title='Loss', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model architecture info
    with st.expander("🏗️ Model Architecture"):
        st.code("""
Model: ResNet-50 (Fine-tuned)
Total Parameters: 23,528,515
Trainable Parameters: 23,528,515
Input Size: 224x224x3
Output Classes: 3 (Cat, Dog, Panda)
Optimizer: Adam (lr=0.001)
Loss Function: CrossEntropyLoss
        """)

# Tab 3: Prediction History
with tab3:
    st.header("📈 Prediction History")
    
    if len(st.session_state.prediction_history) > 0:
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        total_predictions = len(st.session_state.prediction_history)
        avg_confidence = np.mean([p['confidence'] for p in st.session_state.prediction_history])
        
        predictions_count = pd.Series([p['prediction'] for p in st.session_state.prediction_history]).value_counts()
        most_common = predictions_count.index[0] if len(predictions_count) > 0 else "N/A"
        
        with col1:
            st.metric("Total Predictions", total_predictions)
        with col2:
            st.metric("Avg Confidence", f"{avg_confidence:.2f}%")
        with col3:
            st.metric("Most Common", most_common)
        
        st.markdown("---")
        
        # Distribution chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction Distribution")
            fig = px.pie(values=predictions_count.values, names=predictions_count.index,
                        hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Confidence Over Time")
            confidence_data = pd.DataFrame(st.session_state.prediction_history)
            fig = px.line(confidence_data, y='confidence', 
                         labels={'index': 'Prediction #', 'confidence': 'Confidence (%)'},
                         markers=True)
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed history table
        st.subheader("Detailed History")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df = history_df[['timestamp', 'prediction', 'confidence']]
        history_df['confidence'] = history_df['confidence'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        # Download history
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download History as CSV",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )
    else:
        st.info("No predictions yet. Upload and classify some images to see history!")
        st.image("https://cdn-icons-png.flaticon.com/512/3237/3237472.png", width=200)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🐾 Pet Classifier v1.0 | Built with Streamlit & PyTorch</p>
    <p>For best results, use high-quality images with clear visibility of the animal</p>
</div>
""", unsafe_allow_html=True)