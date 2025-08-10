"""
Fish Classification Streamlit Web Application
Enhanced with modern UI design and glassmorphism effects
Modified for CNN_Scratch_best.h5 and MobileNet_best.h5 only
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import tensorflow as tf
import json
import os
from datetime import datetime
import warnings
import time
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🐟 AI Fish Classifier",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS with glassmorphism and contemporary design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root Variables for Dark/Light Mode */
    :root {
        --primary-bg: #0F0F23;
        --secondary-bg: #1A1A2E;
        --glass-bg: rgba(255, 255, 255, 0.1);
        --glass-border: rgba(255, 255, 255, 0.2);
        --text-primary: #FFFFFF;
        --text-secondary: #B0B0C4;
        --accent-blue: #00D4FF;
        --accent-purple: #8B5CF6;
        --accent-green: #10B981;
        --accent-pink: #F472B6;
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-accent: linear-gradient(135deg, #00D4FF 0%, #8B5CF6 100%);
        --shadow-glass: 0 8px 32px rgba(31, 38, 135, 0.37);
        --shadow-elevated: 0 25px 50px rgba(0, 0, 0, 0.25);
    }
    
    /* Global Styles */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #0F0F23 0%, #1A1A2E 50%, #16213E 100%);
        color: var(--text-primary);
        min-height: 100vh;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0F0F23 0%, #1A1A2E 50%, #16213E 100%);
    }
    
    /* Header Styles */
    .modern-header {
        text-align: center;
        padding: 3rem 0;
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: clamp(2.5rem, 5vw, 4rem);
        font-weight: 700;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        font-size: 1.25rem;
        color: var(--text-secondary);
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Glass Card Components */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: var(--shadow-glass);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-elevated);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
    }
    
    /* Prediction Results */
    .prediction-hero {
        background: var(--gradient-accent);
        border-radius: 24px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
        animation: slideInUp 0.6s ease-out;
    }
    
    .prediction-hero::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Ccircle cx='30' cy='30' r='4'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        animation: float 6s ease-in-out infinite;
    }
    
    .prediction-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .prediction-confidence {
        font-size: 1.5rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.9);
        position: relative;
        z-index: 1;
    }
    
    .prediction-model {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.7);
        margin-top: 1rem;
        position: relative;
        z-index: 1;
    }
    
    /* Success States */
    .success-card {
        background: linear-gradient(135deg, var(--accent-green), #059669);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        color: white;
        animation: pulse 2s infinite;
        box-shadow: 0 20px 40px rgba(16, 185, 129, 0.3);
    }
    
    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: var(--glass-bg);
        backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: var(--accent-blue);
        box-shadow: 0 12px 24px rgba(0, 212, 255, 0.2);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--accent-blue);
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    /* Modern Buttons */
    .stButton > button {
        background: var(--gradient-accent);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 16px rgba(0, 212, 255, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 24px rgba(0, 212, 255, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Confidence Bars */
    .confidence-container {
        margin: 2rem 0;
    }
    
    .confidence-item {
        background: var(--glass-bg);
        backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.75rem 0;
        animation: slideInRight 0.5s ease-out;
        transition: all 0.3s ease;
    }
    
    .confidence-item:hover {
        border-color: var(--accent-purple);
        transform: translateX(4px);
    }
    
    .confidence-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }
    
    .confidence-label {
        font-weight: 600;
        color: var(--text-primary);
        font-size: 0.95rem;
    }
    
    .confidence-value {
        font-weight: 700;
        color: var(--accent-blue);
        font-size: 0.95rem;
    }
    
    .progress-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        overflow: hidden;
        height: 8px;
        position: relative;
    }
    
    .progress-bar {
        height: 100%;
        background: var(--gradient-accent);
        border-radius: 8px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .progress-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255, 255, 255, 0.3) 50%, transparent 70%);
        animation: shimmer 2s infinite;
    }
    
    /* Image Container */
    .image-container {
        border-radius: 20px;
        overflow: hidden;
        background: var(--glass-bg);
        backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border);
        padding: 1rem;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .image-container:hover {
        transform: scale(1.02);
        border-color: var(--accent-blue);
    }
    
    .image-container img {
        border-radius: 12px;
    }
    
    /* Sidebar Styling */
    .stSidebar {
        background: linear-gradient(180deg, var(--primary-bg) 0%, var(--secondary-bg) 100%);
    }
    
    .stSidebar .stSelectbox > div > div {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
    }
    
    .stSidebar .stSlider > div > div > div {
        background: var(--accent-blue);
    }
    
    /* Info Cards */
    .info-card {
        background: var(--glass-bg);
        backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid var(--accent-blue);
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        border-left-color: var(--accent-purple);
        transform: translateX(4px);
    }
    
    /* Animations */
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(40px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(40px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% { 
            box-shadow: 0 20px 40px rgba(16, 185, 129, 0.3);
        }
        50% { 
            box-shadow: 0 25px 50px rgba(16, 185, 129, 0.5);
        }
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        color: var(--text-secondary);
        padding: 0.75rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--gradient-accent);
        color: white;
        border-color: transparent;
    }
    
    /* Footer */
    .modern-footer {
        background: var(--glass-bg);
        backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin-top: 4rem;
        position: relative;
        overflow: hidden;
    }
    
    .modern-footer::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--gradient-accent);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .glass-card {
            padding: 1.5rem;
            border-radius: 16px;
        }
        
        .prediction-hero {
            padding: 2rem 1.5rem;
        }
        
        .main-title {
            font-size: 2.5rem;
        }
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--primary-bg);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--glass-border);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-blue);
    }
    
    /* Dataframe Styling */
    .stDataFrame > div {
        background: var(--glass-bg);
        backdrop-filter: blur(16px);
        border-radius: 16px;
        border: 1px solid var(--glass-border);
    }
</style>
""", unsafe_allow_html=True)

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
MAX_FILE_SIZE = 10  # MB

# Specific models to load
TARGET_MODELS = ['CNN_Scratch_best.h5', 'MobileNet_best.h5']

@st.cache_resource
def load_models_and_info():
    """Load only CNN_Scratch and MobileNet models with error handling"""
    models = {}
    results_info = None
    class_names = []
    
    try:
        # Try multiple possible base directories
        possible_base_dirs = [
            os.path.abspath('.'),      # Current directory (for cloud deployment)
            os.path.abspath('..'),     # Parent directory (for local streamlit_app folder)
        ]
        
        for base_dir in possible_base_dirs:
            # Try compressed models first, then original models
            possible_model_dirs = [
                os.path.join(base_dir, 'results', 'compressed_models'),  # Compressed models (preferred)
                os.path.join(base_dir, 'results', 'models'),             # Original models (backup)
            ]
            
            for models_dir in possible_model_dirs:
                if os.path.exists(models_dir):
                    # Look specifically for our target models
                    available_files = os.listdir(models_dir)
                    
                    # Check for exact matches or compressed versions
                    model_files_to_load = []
                    for target_model in TARGET_MODELS:
                        # Check for exact match
                        if target_model in available_files:
                            model_files_to_load.append(target_model)
                        # Check for compressed version
                        compressed_name = target_model.replace('.h5', '_compressed.h5')
                        if compressed_name in available_files:
                            model_files_to_load.append(compressed_name)
                    
                    if model_files_to_load:
                        st.info(f"📁 Loading models from: {os.path.basename(models_dir)}")
                        
                        for model_file in model_files_to_load:
                            # Clean model name for display
                            if 'CNN_Scratch' in model_file:
                                model_name = 'CNN_Scratch'
                                display_name = '🧠 Custom CNN (From Scratch)'
                            elif 'MobileNet' in model_file:
                                model_name = 'MobileNet'
                                display_name = '📱 MobileNet (Transfer Learning)'
                            else:
                                continue
                            
                            model_path = os.path.join(models_dir, model_file)
                            
                            try:
                                # Check file size
                                file_size = os.path.getsize(model_path) / (1024 * 1024)
                                
                                # Load model
                                model = tf.keras.models.load_model(model_path)
                                
                                models[model_name] = {
                                    'model': model,
                                    'path': model_path,
                                    'file': model_file,
                                    'size_mb': file_size,
                                    'display_name': display_name
                                }
                                
                                # Show success message with size info
                                compressed_indicator = "🗜️ " if "_compressed" in model_file else ""
                                st.success(f"✅ {compressed_indicator}Loaded {display_name} ({file_size:.1f}MB)")
                                
                            except Exception as e:
                                st.warning(f"Could not load {model_name}: {str(e)}")
                        
                        # If we found our target models, break out of the loops
                        if len(models) >= 2:
                            break
            
            # If we found models, break out of base_dir loop
            if models:
                break
        
        # Load results JSON if available
        for base_dir in possible_base_dirs:
            results_file = os.path.join(base_dir, 'results', 'combined_training_results.json')
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        results_info = json.load(f)
                    class_names = results_info.get('dataset_info', {}).get('class_names', [])
                    break
                except Exception as e:
                    st.warning(f"Could not load results file: {e}")
        
        # If no class names from results, try to infer from training data
        if not class_names:
            for base_dir in possible_base_dirs:
                train_dir = os.path.join(base_dir, 'data', 'train')
                if os.path.exists(train_dir):
                    class_names = [d for d in os.listdir(train_dir) 
                                 if os.path.isdir(os.path.join(train_dir, d))]
                    class_names.sort()
                    break
        
        return models, results_info, class_names
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}, None, []

def preprocess_image(image, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Preprocess uploaded image for prediction"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(target_size)
        img_array = np.array(image)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_image(model, image_array, class_names):
    """Make prediction on preprocessed image"""
    try:
        predictions = model.predict(image_array, verbose=0)
        class_probabilities = predictions[0]
        top_indices = np.argsort(class_probabilities)[::-1]
        
        results = []
        for i in range(min(5, len(class_names) if class_names else len(class_probabilities))):
            if i < len(top_indices):
                idx = top_indices[i]
                if idx < len(class_names) and class_names:
                    results.append({
                        'class': class_names[idx],
                        'confidence': float(class_probabilities[idx]),
                        'percentage': f"{class_probabilities[idx] * 100:.2f}%"
                    })
                else:
                    results.append({
                        'class': f'Class_{idx}',
                        'confidence': float(class_probabilities[idx]),
                        'percentage': f"{class_probabilities[idx] * 100:.2f}%"
                    })
        return results
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def create_modern_chart(predictions):
    """Create modern matplotlib chart with dark theme"""
    if not predictions:
        return None
    
    # Set dark style
    plt.style.use('dark_background')
    
    classes = [pred['class'][:20] + '...' if len(pred['class']) > 20 else pred['class'] 
              for pred in predictions[:5]]
    confidences = [pred['confidence'] for pred in predictions[:5]]
    
    # Create figure with custom styling
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1A1A2E')
    ax.set_facecolor('#1A1A2E')
    
    # Create horizontal bar chart with modern colors
    colors = ['#00D4FF', '#8B5CF6', '#F472B6', '#10B981', '#F59E0B']
    bars = ax.barh(classes, confidences, color=colors[:len(classes)], height=0.6, alpha=0.9)
    
    # Add value labels with modern styling
    for i, (bar, conf) in enumerate(zip(bars, confidences)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{conf:.1%}', ha='left', va='center', fontweight='600',
               fontsize=12, color='#FFFFFF')
    
    # Customize chart appearance
    ax.set_xlabel('Confidence Score', fontsize=14, fontweight='600', color='#FFFFFF')
    ax.set_title('🐟 Fish Species Classification Results', fontsize=18, fontweight='700', 
                color='#FFFFFF', pad=20)
    ax.set_xlim(0, max(confidences) * 1.2 if confidences else 1)
    
    # Remove spines and customize grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444')
    ax.spines['bottom'].set_color('#444')
    ax.grid(True, alpha=0.2, linestyle='--', color='#666')
    
    # Customize tick labels
    ax.tick_params(colors='#FFFFFF', labelsize=11)
    
    plt.tight_layout()
    return fig

def enhance_image(image, brightness=1.0, contrast=1.0, sharpness=1.0):
    """Apply image enhancements"""
    try:
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(sharpness)
        return image
    except Exception as e:
        st.error(f"Error enhancing image: {str(e)}")
        return image

def display_modern_confidence_bars(predictions):
    """Display modern confidence bars with glassmorphism"""
    st.markdown('<div class="confidence-container">', unsafe_allow_html=True)
    st.markdown("### 🎯 Prediction Confidence")
    
    for i, pred in enumerate(predictions[:5]):
        confidence = pred['confidence']
        class_name = pred['class'].replace('_', ' ').title()
        
        # Create modern progress bar HTML
        confidence_html = f"""
        <div class="confidence-item" style="animation-delay: {i * 0.1}s;">
            <div class="confidence-header">
                <span class="confidence-label">{class_name}</span>
                <span class="confidence-value">{pred['percentage']}</span>
            </div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {confidence * 100}%;"></div>
            </div>
        </div>
        """
        st.markdown(confidence_html, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Modern Header
    st.markdown("""
    <div class="modern-header">
        <h1 class="main-title">🐟 AI Fish Species Classifier</h1>
        <p class="subtitle">Comparing Custom CNN vs Transfer Learning Models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models with loading animation
    with st.spinner("🧠 Loading AI models..."):
        models, results_info, class_names = load_models_and_info()
    
    if len(models) < 2:
        st.markdown("""
        <div class="glass-card">
            <h2>⚠️ Models Not Found</h2>
            <p>Looking for CNN_Scratch_best.h5 and MobileNet_best.h5</p>
            <p>Please ensure these models are in your results/models/ directory.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📋 Expected Models"):
            st.code("""
Required models:
• CNN_Scratch_best.h5 (Custom CNN from scratch)
• MobileNet_best.h5 (Transfer learning model)

Location: results/models/ or results/compressed_models/
            """)
        return
    
    # Success message for loaded models
    st.markdown(f"""
    <div class="success-card">
        <h3>✅ AI Models Ready!</h3>
        <p>Successfully loaded {len(models)} specialized models</p>
        <p>🧠 Custom CNN vs 📱 MobileNet Transfer Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # Model Selection
        st.markdown("### 🤖 AI Model Selection")
        
        # Create a more descriptive model selection
        model_options = {}
        for model_name, model_data in models.items():
            display_name = model_data.get('display_name', model_name)
            model_options[display_name] = model_name
        
        selected_display_name = st.selectbox(
            "Choose your AI model:",
            list(model_options.keys()),
            help="Compare performance between custom CNN and transfer learning"
        )
        
        selected_model = model_options[selected_display_name]
        
        # Display model performance with modern cards
        if results_info and selected_model in results_info.get('training_results', {}):
            model_info = results_info['training_results'][selected_model]
            if model_info.get('status') == 'SUCCESS':
                accuracy = model_info.get('best_val_accuracy', 0)
                top3_acc = model_info.get('best_val_top3_accuracy', 0)
                training_time = model_info.get('training_time_minutes', 0)
                
                st.markdown(f"""
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{accuracy:.1%}</div>
                        <div class="metric-label">Accuracy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{top3_acc:.1%}</div>
                        <div class="metric-label">Top-3 Accuracy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{training_time:.1f}m</div>
                        <div class="metric-label">Training Time</div>
                    </div>
                </div>
