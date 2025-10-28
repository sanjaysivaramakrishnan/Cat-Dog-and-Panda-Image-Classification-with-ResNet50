import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import json
import os
import requests

# 🔗 Your Hugging Face model file URL
MODEL_URL = "https://huggingface.co/SanjaySivaramakrishnan/animal-classification-model/resolve/main/model.pth"

# 📥 Auto-download model if not present
def download_model():
    if not os.path.exists('model.pth'):
        try:
            st.info("📥 Downloading model weights from Hugging Face...")
            response = requests.get(MODEL_URL)
            if response.status_code == 200:
                with open('model.pth', 'wb') as f:
                    f.write(response.content)
                st.success("✅ Model downloaded successfully!")
            else:
                st.error(f"Failed to download model (status code {response.status_code})")
        except Exception as e:
            st.error(f"Error downloading model: {str(e)}")

# 🔧 Download model before loading
download_model()

# Page configuration
st.set_page_config(
    page_title="Animal Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        font-size: 1.1rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained ResNet50 model with proper error handling"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists('model.pth'):
        st.error("⚠️ Model file not found even after download attempt.")
        return None, device

    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.7),
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 3)
    )

    try:
        model.load_state_dict(torch.load('model.pth', map_location=device))
        model = model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, device

@st.cache_data
def load_metrics():
    if not os.path.exists('metrics.json'):
        return None
    try:
        with open('metrics.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading metrics: {str(e)}")
        return None

def get_transforms():
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def predict_image(image, model, device):
    transform = get_transforms()
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item(), probabilities[0].cpu().numpy()

def create_confidence_gauge(confidence, class_name):
    """Create a gauge chart for confidence"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': f"Confidence: {class_name}"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=250)
    return fig

def create_probability_chart(probabilities, class_names):
    """Create horizontal bar chart for probabilities"""
    fig = go.Figure(go.Bar(
        x=probabilities * 100,
        y=class_names,
        orientation='h',
        marker=dict(
            color=probabilities * 100,
            colorscale='Viridis',
            showscale=False
        ),
        text=[f"{p*100:.2f}%" for p in probabilities],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Prediction Probabilities',
        xaxis_title='Probability (%)',
        yaxis_title='Class',
        height=300,
        xaxis_range=[0, 100]
    )
    return fig

def create_confusion_matrix_plot(cm, class_names):
    """Create interactive confusion matrix"""
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations with both count and percentage
    annotations = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            annotations.append(f"{cm[i, j]}<br>({cm_percent[i, j]:.1f}%)")
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=np.array(annotations).reshape(cm.shape),
        texttemplate='%{text}',
        textfont={"size": 14},
        showscale=True,
        colorbar=dict(title="Count")
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=600,
        height=500
    )
    
    return fig

def create_metrics_bar_chart(metrics_df):
    """Create grouped bar chart for metrics"""
    fig = go.Figure()
    
    metrics = ['precision', 'recall', 'f1-score']
    colors = ['#667eea', '#764ba2', '#f093fb']
    
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric.capitalize(),
            x=metrics_df['class'],
            y=metrics_df[metric],
            marker_color=colors[i],
            text=[f"{v:.3f}" for v in metrics_df[metric]],
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Classification Metrics by Class',
        xaxis_title='Class',
        yaxis_title='Score',
        barmode='group',
        yaxis_range=[0, 1.1],
        height=500,
        legend=dict(x=0.7, y=1.0)
    )
    
    return fig

def create_training_plots(history_data):
    """Create combined training history plots"""
    epochs = list(range(1, len(history_data['train_loss']) + 1))
    
    # Create subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Loss', 'Accuracy')
    )
    
    # Loss traces
    fig.add_trace(
        go.Scatter(x=epochs, y=history_data['train_loss'],
                  mode='lines+markers', name='Train Loss',
                  line=dict(color='#667eea', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history_data['val_loss'],
                  mode='lines+markers', name='Val Loss',
                  line=dict(color='#f093fb', width=2)),
        row=1, col=1
    )
    
    # Accuracy traces
    fig.add_trace(
        go.Scatter(x=epochs, y=history_data['train_acc'],
                  mode='lines+markers', name='Train Acc',
                  line=dict(color='#4CAF50', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history_data['val_acc'],
                  mode='lines+markers', name='Val Acc',
                  line=dict(color='#FF6B6B', width=2)),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=True, hovermode='x unified')
    
    return fig

def prediction_page():
    """Main prediction interface"""
    st.title("🐾 Animal Classifier")
    st.markdown("### Classify images as **Cat**, **Dog**, or **Panda**")
    st.markdown("---")
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.error("⚠️ Model file 'model.pth' not found!")
        st.info("""
        💡 **Setup Instructions:**
        1. Train your model using the training script
        2. Ensure 'model.pth' is in the same directory as this app
        3. Restart the Streamlit app
        """)
        return
    
    # Class configuration
    class_names = ['Cat', 'Dog', 'Panda']
    class_emojis = ['🐱', '🐕', '🐼']
    
    # Main layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload a clear image of a cat, dog, or panda"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_container_width=True)
                
                # Image info
                st.caption(f"Image size: {image.size[0]}x{image.size[1]} pixels")
                
                # Predict button
                if st.button("🔍 Classify Image", use_container_width=True):
                    with st.spinner('🔄 Analyzing image...'):
                        pred_idx, confidence, probabilities = predict_image(image, model, device)
                        
                        # Store in session state
                        st.session_state.prediction = {
                            'class': class_names[pred_idx],
                            'emoji': class_emojis[pred_idx],
                            'confidence': confidence,
                            'probabilities': probabilities,
                            'class_idx': pred_idx
                        }
                        st.success("✅ Classification complete!")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    with col2:
        st.subheader("📊 Prediction Results")
        
        if 'prediction' in st.session_state:
            pred = st.session_state.prediction
            
            # Main prediction display
            st.markdown(f"""
                <div class="prediction-box">
                    <h1 style='text-align: center; font-size: 4rem;'>{pred['emoji']}</h1>
                    <h2 style='text-align: center; color: #1f77b4;'>{pred['class']}</h2>
                    <p style='text-align: center; font-size: 1.3rem; margin-top: 1rem;'>
                        Confidence: <strong style='color: {"#4CAF50" if pred["confidence"] > 0.9 else "#FF9800"};'>
                        {pred['confidence']*100:.2f}%</strong>
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Probability chart
            st.plotly_chart(
                create_probability_chart(pred['probabilities'], class_names),
                use_container_width=True
            )
            
            # Detailed probabilities
            with st.expander("📈 View Detailed Probabilities"):
                for i, (name, emoji) in enumerate(zip(class_names, class_emojis)):
                    prob = pred['probabilities'][i] * 100
                    st.write(f"{emoji} **{name}**: {prob:.4f}%")
        else:
            st.info("👆 Upload an image and click 'Classify Image' to see predictions")
    
    # Model information
    st.markdown("---")
    st.markdown("### ℹ️ Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🏗️ Architecture**")
        st.write("• ResNet50 (Transfer Learning)")
        st.write("• Fine-tuned Layer4 + FC")
        st.write("• 3-class classifier")
    
    with col2:
        st.markdown("**📊 Classes**")
        for emoji, name in zip(class_emojis, class_names):
            st.write(f"• {emoji} {name}")
    
    with col3:
        st.markdown("**⚙️ Runtime**")
        st.write(f"• Device: **{device.type.upper()}**")
        if torch.cuda.is_available():
            st.write(f"• GPU: {torch.cuda.get_device_name(0)}")
        st.write(f"• PyTorch: {torch.__version__}")

def metrics_page():
    """Display model performance metrics"""
    st.title("📊 Model Performance Metrics")
    st.markdown("### Comprehensive analysis of model performance")
    st.markdown("---")
    
    # Load metrics
    metrics_data = load_metrics()
    
    if metrics_data is None:
        st.error("❌ Metrics file 'metrics.json' not found!")
        st.info("Train your model first to generate metrics.")
        return
    
    class_names = metrics_data.get('class', ['Cat', 'Dog', 'Panda'])
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview", "🎯 Confusion Matrix", 
        "📉 Training History", "📝 Detailed Report"
    ])
    
    with tab1:
        st.subheader("Model Performance Overview")
        
        # Key metrics
        test_acc = metrics_data.get('test_accuracy', 0) * 100
        macro_f1 = metrics_data.get('Macro_Avg', {}).get('f1-score', 0) * 100
        weighted_precision = metrics_data.get('Weighted_Avg', {}).get('precision', 0) * 100
        weighted_recall = metrics_data.get('Weighted_Avg', {}).get('recall', 0) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h2>{test_acc:.2f}%</h2>
                    <p>Test Accuracy</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <h2>{macro_f1:.2f}%</h2>
                    <p>Macro F1-Score</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                    <h2>{weighted_precision:.2f}%</h2>
                    <p>Weighted Precision</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                    <h2>{weighted_recall:.2f}%</h2>
                    <p>Weighted Recall</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Per-class metrics
        st.subheader("Per-Class Performance")
        
        metrics_df = pd.DataFrame({
            'Class': metrics_data['class'],
            'Precision': metrics_data['precision'],
            'Recall': metrics_data['recall'],
            'F1-Score': metrics_data['f1-score'],
            'Support': metrics_data['support']
        })
        
        # Style the dataframe
        st.dataframe(
            metrics_df.style.format({
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1-Score': '{:.4f}',
                'Support': '{:.0f}'
            }).background_gradient(cmap='Greens', subset=['Precision', 'Recall', 'F1-Score']),
            use_container_width=True,
            hide_index=True
        )
        
        # Metrics bar chart
        st.plotly_chart(
            create_metrics_bar_chart(pd.DataFrame({
                'class': metrics_data['class'],
                'precision': metrics_data['precision'],
                'recall': metrics_data['recall'],
                'f1-score': metrics_data['f1-score']
            })),
            use_container_width=True
        )
    
    with tab2:
        st.subheader("Confusion Matrix Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        cm = np.array(metrics_data['confusion_matrix'])
        
        with col1:
            st.plotly_chart(
                create_confusion_matrix_plot(cm, class_names),
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### 📖 Interpretation")
            st.write("**Diagonal values:** Correct predictions")
            st.write("**Off-diagonal:** Misclassifications")
            
            st.markdown("---")
            st.markdown("#### 📊 Summary")
            
            total_correct = np.trace(cm)
            total_samples = np.sum(cm)
            accuracy = total_correct / total_samples
            
            st.metric("Accuracy", f"{accuracy*100:.2f}%")
            st.metric("Total Samples", f"{total_samples:,}")
            st.metric("Correct", f"{total_correct:,}")
            st.metric("Incorrect", f"{total_samples - total_correct:,}")
            
            # Per-class accuracy
            st.markdown("---")
            st.markdown("#### Per-Class Accuracy")
            for i, class_name in enumerate(class_names):
                class_acc = cm[i, i] / cm[i].sum() * 100
                st.write(f"**{class_name}:** {class_acc:.2f}%")
    
    with tab3:
        st.subheader("Training History")
        
        history_data = metrics_data.get('history', {})
        
        if history_data:
            # Combined plot
            st.plotly_chart(
                create_training_plots(history_data),
                use_container_width=True
            )
            
            # Training summary
            st.markdown("---")
            st.markdown("#### 📊 Training Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Final Metrics:**")
                final_train_loss = history_data['train_loss'][-1]
                final_val_loss = history_data['val_loss'][-1]
                final_train_acc = history_data['train_acc'][-1]
                final_val_acc = history_data['val_acc'][-1]
                
                st.write(f"• Train Loss: {final_train_loss:.4f}")
                st.write(f"• Val Loss: {final_val_loss:.4f}")
                st.write(f"• Train Accuracy: {final_train_acc*100:.2f}%")
                st.write(f"• Val Accuracy: {final_val_acc*100:.2f}%")
            
            with col2:
                st.markdown("**Best Metrics:**")
                best_val_acc = max(history_data['val_acc'])
                best_val_acc_epoch = history_data['val_acc'].index(best_val_acc) + 1
                min_val_loss = min(history_data['val_loss'])
                min_val_loss_epoch = history_data['val_loss'].index(min_val_loss) + 1
                
                st.write(f"• Best Val Acc: {best_val_acc*100:.2f}% (Epoch {best_val_acc_epoch})")
                st.write(f"• Min Val Loss: {min_val_loss:.4f} (Epoch {min_val_loss_epoch})")
                st.write(f"• Total Epochs: {len(history_data['train_loss'])}")
            
            # Key observations
            st.markdown("---")
            st.markdown("#### 📝 Key Observations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Check for overfitting
                acc_gap = (final_train_acc - final_val_acc) * 100
                if acc_gap < 5:
                    st.success("✅ Minimal overfitting detected")
                elif acc_gap < 10:
                    st.info("ℹ️ Slight overfitting present")
                else:
                    st.warning("⚠️ Notable overfitting detected")
                
                # Check convergence
                if len(history_data['val_loss']) > 5:
                    recent_loss_std = np.std(history_data['val_loss'][-5:])
                    if recent_loss_std < 0.01:
                        st.success("✅ Model converged well")
                    else:
                        st.info("ℹ️ Model still improving")
            
            with col2:
                st.info(f"ℹ️ Training stopped at epoch {len(history_data['train_loss'])}")
                
                # Check if early stopping was used
                if len(history_data['train_loss']) < 20:
                    st.info("ℹ️ Early stopping likely prevented overfitting")
        else:
            st.warning("⚠️ No training history data available")
    
    with tab4:
        st.subheader("Detailed Classification Report")
        
        # Full metrics table
        report_df = pd.DataFrame({
            'Class': metrics_data['class'],
            'Precision': metrics_data['precision'],
            'Recall': metrics_data['recall'],
            'F1-Score': metrics_data['f1-score'],
            'Support': metrics_data['support']
        })
        
        # Add average rows
        macro_avg = pd.DataFrame({
            'Class': ['Macro Avg'],
            'Precision': [metrics_data['Macro_Avg']['precision']],
            'Recall': [metrics_data['Macro_Avg']['recall']],
            'F1-Score': [metrics_data['Macro_Avg']['f1-score']],
            'Support': [sum(metrics_data['support'])]
        })
        
        weighted_avg = pd.DataFrame({
            'Class': ['Weighted Avg'],
            'Precision': [metrics_data['Weighted_Avg']['precision']],
            'Recall': [metrics_data['Weighted_Avg']['recall']],
            'F1-Score': [metrics_data['Weighted_Avg']['f1-score']],
            'Support': [sum(metrics_data['support'])]
        })
        
        full_report_df = pd.concat([report_df, macro_avg, weighted_avg], ignore_index=True)
        
        st.dataframe(
            full_report_df.style.format({
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1-Score': '{:.4f}',
                'Support': '{:.0f}'
            }).apply(lambda x: ['background-color: #f0f2f6' if i >= len(report_df) 
                               else '' for i in range(len(x))], axis=0),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        
        # Model details
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏗️ Model Architecture")
            st.markdown("""
            **Base Model:**
            - ResNet50 (Pre-trained on ImageNet)
            - Transfer Learning approach
            
            **Fine-tuned Layers:**
            - Layer4 (deeper convolutional layers)
            - Fully Connected layers
            
            **Custom Classifier:**
            - Linear(2048 → 512) + ReLU + Dropout(0.7)
            - Linear(512 → 128) + ReLU + Dropout(0.3)
            - Linear(128 → 3) [Output layer]
            
            **Total Parameters:** ~24M
            **Trainable Parameters:** ~16M
            """)
        
        with col2:
            st.markdown("#### ⚙️ Training Configuration")
            st.markdown("""
            **Optimizer:**
            - AdamW optimizer
            - Learning Rate: 5e-6
            - Weight Decay: 5e-4
            
            **Loss Function:**
            - CrossEntropyLoss
            - Label Smoothing: 0.1
            
            **Training Settings:**
            - Batch Size: 32
            - Max Epochs: 20
            - Early Stopping: Patience 5
            
            **Data Augmentation:**
            - Random Resize Crop (scale: 0.8-1.0)
            - Random Horizontal Flip
            - Random Rotation (±10°)
            - Color Jitter (brightness, contrast, saturation)
            """)
        
        # Download metrics
        st.markdown("---")
        st.markdown("#### 💾 Export Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download full report as CSV
            csv = full_report_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Report (CSV)",
                data=csv,
                file_name="model_metrics_report.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download metrics as JSON
            json_str = json.dumps(metrics_data, indent=2)
            st.download_button(
                label="📥 Download Metrics (JSON)",
                data=json_str,
                file_name="metrics.json",
                mime="application/json"
            )

def main():
    """Main application"""
    
    # Sidebar
    st.sidebar.title("🐾 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select Page:",
        ["🔮 Prediction", "📊 Model Metrics"],
        label_visibility="visible"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        """
        This application uses a fine-tuned **ResNet50** model 
        to classify images of cats, dogs, and pandas with high accuracy.
        
        The model was trained using transfer learning on a custom dataset.
        """
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Model Specifications")
    st.sidebar.markdown("""
    - **Architecture:** ResNet50
    - **Classes:** Cat, Dog, Panda
    - **Framework:** PyTorch
    - **Input Size:** 224×224
    - **Preprocessing:** ImageNet normalization
    """)
    
    # Check file availability
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 File Status")
    
    model_exists = os.path.exists('model.pth')
    metrics_exists = os.path.exists('metrics.json')
    
    st.sidebar.write(f"{'✅' if model_exists else '❌'} model.pth")
    st.sidebar.write(f"{'✅' if metrics_exists else '❌'} metrics.json")
    
    # Route to page
    if page == "🔮 Prediction":
        prediction_page()
    else:
        metrics_page()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <p style='text-align: center; color: gray; font-size: 0.9rem;'>
        Built by Sanjay using Streamlit and PyTorch | 
        ResNet50 Transfer Learning Model
        </p>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()