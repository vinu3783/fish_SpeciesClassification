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
        color: var(--text-primary);
    }
    
    /* Streamlit Component Overrides */
    .stMarkdown, .stMarkdown p, .stMarkdown div {
        color: var(--text-primary) !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: var(--text-primary) !important;
    }
    
    .stSelectbox label, .stSlider label, .stFileUploader label {
        color: var(--text-primary) !important;
    }
    
    .stExpander .streamlit-expanderHeader {
        color: var(--text-primary) !important;
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
        color: var(--text-primary);
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
    
    .glass-card h1, .glass-card h2, .glass-card h3, .glass-card h4 {
        color: var(--text-primary) !important;
    }
    
    .glass-card p, .glass-card ul, .glass-card li {
        color: var(--text-primary) !important;
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
        color: var(--text-primary) !important;
    }
    
    .stSidebar .stSlider > div > div > div {
        background: var(--accent-blue);
    }
    
    .stSidebar .stMarkdown, .stSidebar .stMarkdown p, .stSidebar .stMarkdown div {
        color: var(--text-primary) !important;
    }
    
    .stSidebar .stMarkdown h1, .stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3 {
        color: var(--text-primary) !important;
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
        color: var(--text-primary);
    }
    
    .info-card:hover {
        border-left-color: var(--accent-purple);
        transform: translateX(4px);
    }
    
    .info-card p, .info-card ul, .info-card li {
        color: var(--text-primary) !important;
    }
    
    .info-card h3, .info-card h4 {
        color: var(--text-primary) !important;
        margin-bottom: 1rem;
    }
    
    .info-card strong {
        color: var(--accent-blue);
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
    
    .stDataFrame table {
        color: var(--text-primary) !important;
    }
    
    .stDataFrame th {
        background-color: var(--secondary-bg) !important;
        color: var(--text-primary) !important;
        border-color: var(--glass-border) !important;
    }
    
    .stDataFrame td {
        background-color: var(--glass-bg) !important;
        color: var(--text-primary) !important;
        border-color: var(--glass-border) !important;
    }
    
    /* File Uploader */
    .stFileUploader > div > div {
        background: var(--glass-bg);
        border: 2px dashed var(--glass-border);
        border-radius: 16px;
        color: var(--text-primary) !important;
    }
    
    /* Expander */
    .stExpander {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
    }
    
    .stExpander .streamlit-expanderContent {
        background: var(--glass-bg);
        color: var(--text-primary) !important;
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
    success_html = f"""
    <div class="success-card">
        <h3>✅ AI Models Ready!</h3>
        <p>Successfully loaded {len(models)} specialized models</p>
        <p>🧠 Custom CNN vs 📱 MobileNet Transfer Learning</p>
    </div>
    """
    st.markdown(success_html, unsafe_allow_html=True)
    
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
                
                metric_html = f"""
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
                """
                st.markdown(metric_html, unsafe_allow_html=True)
        
        # Show model file info
        if selected_model in models:
            model_data = models[selected_model]
            info_html = f"""
            <div class="info-card">
                <p><strong>📄 File:</strong> {model_data['file']}</p>
                <p><strong>📊 Size:</strong> {model_data['size_mb']:.1f} MB</p>
            </div>
            """
            st.markdown(info_html, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Image Enhancement Controls
        st.markdown("### 🎨 Image Enhancement")
        st.markdown("*Adjust these settings to improve recognition accuracy*")
        
        brightness = st.slider("☀️ Brightness", 0.5, 2.0, 1.0, 0.1)
        contrast = st.slider("⚡ Contrast", 0.5, 2.0, 1.0, 0.1)
        sharpness = st.slider("🔍 Sharpness", 0.0, 2.0, 1.0, 0.1)
        
        st.markdown("---")
        
        # App Statistics
        st.markdown("### 📊 Model Comparison")
        stats_html = f"""
        <div class="info-card">
            <p><strong>🧠 Custom CNN:</strong> Built from scratch</p>
            <p><strong>📱 MobileNet:</strong> Transfer learning</p>
            <p><strong>🏷️ Fish Species:</strong> {len(class_names) if class_names else 'Unknown'}</p>
            <p><strong>🖼️ Max File Size:</strong> {MAX_FILE_SIZE}MB</p>
        </div>
        """
        st.markdown(stats_html, unsafe_allow_html=True)
    
    # Main Content Area with improved layout
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h2>📤 Upload Your Fish Image</h2>
        """, unsafe_allow_html=True)
        
        # Beautiful file uploader
        uploaded_file = st.file_uploader(
            "Choose a fish image...",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help=f"Supported formats: PNG, JPG, JPEG, BMP (Max: {MAX_FILE_SIZE}MB)"
        )
        
        if uploaded_file is not None:
            # File size check
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            if file_size > MAX_FILE_SIZE:
                st.error(f"❌ File too large! Maximum: {MAX_FILE_SIZE}MB (Current: {file_size:.1f}MB)")
                return
            
            # Load and display image
            try:
                image = Image.open(uploaded_file)
                enhanced_image = enhance_image(image, brightness, contrast, sharpness)
                
                # Beautiful image display with tabs
                tab1, tab2 = st.tabs(["✨ Enhanced", "📷 Original"])
                
                with tab1:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(enhanced_image, caption="Enhanced for AI Analysis", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab2:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(image, caption=f"Original: {uploaded_file.name}", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Image info card
                image_info_html = f"""
                <div class="info-card">
                    <p><strong>📊 Image Details:</strong></p>
                    <p>• Size: {image.size[0]} × {image.size[1]} pixels</p>
                    <p>• Mode: {image.mode}</p>
                    <p>• File size: {file_size:.2f} MB</p>
                </div>
                """
                st.markdown(image_info_html, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
                return
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h2>🤖 AI Analysis Results</h2>
        """, unsafe_allow_html=True)
        
        if uploaded_file is not None and models and class_names:
            
            # Beautiful prediction button
            if st.button("🔍 Analyze Fish Species", type="primary", use_container_width=True):
                
                # Loading animation
                with st.spinner("🧠 AI is analyzing your image..."):
                    time.sleep(1)  # Add slight delay for better UX
                    
                    try:
                        processed_image = preprocess_image(enhanced_image)
                        
                        if processed_image is not None:
                            model_data = models[selected_model]
                            model = model_data['model']
                            
                            predictions = predict_image(model, processed_image, class_names)
                            
                            if predictions:
                                top_prediction = predictions[0]
                                
                                # Beautiful main result card
                                prediction_html = f"""
                                <div class="prediction-hero">
                                    <div class="prediction-title">{top_prediction['class'].replace('_', ' ').title()}</div>
                                    <div class="prediction-confidence">{top_prediction['percentage']}</div>
                                    <div class="prediction-model">{model_data.get('display_name', selected_model)}</div>
                                </div>
                                """
                                st.markdown(prediction_html, unsafe_allow_html=True)
                                
                                # Beautiful confidence bars
                                display_modern_confidence_bars(predictions)
                                
                                # Chart visualization
                                st.markdown("### 📈 Visual Analysis")
                                confidence_chart = create_modern_chart(predictions)
                                if confidence_chart:
                                    st.pyplot(confidence_chart, use_container_width=True)
                                
                                # Export functionality
                                st.markdown("### 💾 Export Results")
                                export_data = {
                                    'timestamp': datetime.now().isoformat(),
                                    'image_name': uploaded_file.name,
                                    'model_used': selected_model,
                                    'model_display_name': model_data.get('display_name', selected_model),
                                    'top_prediction': top_prediction,
                                    'all_predictions': predictions[:5]
                                }
                                
                                json_str = json.dumps(export_data, indent=2)
                                st.download_button(
                                    label="📄 Download Analysis Report",
                                    data=json_str,
                                    file_name=f"fish_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                        else:
                            st.error("❌ Error processing image for prediction")
                    
                    except Exception as e:
                        st.error(f"❌ Analysis error: {str(e)}")
        else:
            # Beautiful instructions card
            st.markdown("""
            <div class="info-card">
                <h3>👆 Ready to Analyze!</h3>
                <p>Upload a fish image using the panel on the left to get started with AI-powered species identification.</p>
                <br>
                <p><strong>💡 Compare Two Approaches:</strong></p>
                <ul>
                    <li><strong>🧠 Custom CNN:</strong> Built specifically for fish classification</li>
                    <li><strong>📱 MobileNet:</strong> Pre-trained model with transfer learning</li>
                </ul>
                <br>
                <p><strong>🎯 Tips for best results:</strong></p>
                <ul>
                    <li>Use clear, well-lit images</li>
                    <li>Ensure the fish is the main subject</li>
                    <li>Try different enhancement settings</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Performance Dashboard - Only show for our 2 models
    if results_info and results_info.get('training_results'):
        st.markdown("---")
        st.markdown("""
        <div class="glass-card">
            <h2>📈 Model Performance Comparison</h2>
        """, unsafe_allow_html=True)
        
        training_results = results_info['training_results']
        # Filter for only our target models
        our_models = {k: v for k, v in training_results.items() 
                     if k in ['CNN_Scratch', 'MobileNet'] and v.get('status') == 'SUCCESS'}
        
        if our_models:
            # Performance metrics table
            performance_data = []
            for model_name, results in our_models.items():
                display_name = "🧠 Custom CNN" if model_name == "CNN_Scratch" else "📱 MobileNet"
                performance_data.append({
                    'Model': display_name,
                    'Type': 'From Scratch' if model_name == "CNN_Scratch" else 'Transfer Learning',
                    'Validation Accuracy': f"{results.get('best_val_accuracy', 0):.1%}",
                    'Top-3 Accuracy': f"{results.get('best_val_top3_accuracy', 0):.1%}",
                    'Training Time': f"{results.get('training_time_minutes', 0):.1f} min",
                    'Parameters': f"{results.get('total_params', 0):,}"
                })
            
            performance_df = pd.DataFrame(performance_data)
            performance_df = performance_df.sort_values('Validation Accuracy', ascending=False)
            
            st.dataframe(performance_df, use_container_width=True, hide_index=True)
            
            # Model comparison insights
            if len(our_models) == 2:
                cnn_results = our_models.get('CNN_Scratch', {})
                mobilenet_results = our_models.get('MobileNet', {})
                
                if cnn_results and mobilenet_results:
                    cnn_acc = cnn_results.get('best_val_accuracy', 0)
                    mobilenet_acc = mobilenet_results.get('best_val_accuracy', 0)
                    cnn_time = cnn_results.get('training_time_minutes', 0)
                    mobilenet_time = mobilenet_results.get('training_time_minutes', 0)
                    
                    st.markdown("### 🔍 Model Analysis")
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        if cnn_acc > mobilenet_acc:
                            winner = "🧠 Custom CNN"
                            diff = ((cnn_acc - mobilenet_acc) / mobilenet_acc) * 100
                        else:
                            winner = "📱 MobileNet"
                            diff = ((mobilenet_acc - cnn_acc) / cnn_acc) * 100
                        
                        winner_html = f"""
                        <div class="info-card">
                            <h4>🏆 Best Accuracy</h4>
                            <p><strong>{winner}</strong></p>
                            <p>{diff:.1f}% better performance</p>
                        </div>
                        """
                        st.markdown(winner_html, unsafe_allow_html=True)
                    
                    with col_b:
                        if cnn_time < mobilenet_time:
                            faster = "🧠 Custom CNN"
                            time_diff = mobilenet_time - cnn_time
                        else:
                            faster = "📱 MobileNet"
                            time_diff = cnn_time - mobilenet_time
                        
                        faster_html = f"""
                        <div class="info-card">
                            <h4>⚡ Faster Training</h4>
                            <p><strong>{faster}</strong></p>
                            <p>{time_diff:.1f} min faster</p>
                        </div>
                        """
                        st.markdown(faster_html, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Modern Footer
    st.markdown("""
    <div class="modern-footer">
        <h3>🐟 AI Fish Species Classifier</h3>
        <p>Comparing Custom CNN vs Transfer Learning Approaches</p>
        <p>🧠 Custom Architecture • 📱 Pre-trained MobileNet</p>
        <br>
        <p>Built with Streamlit & TensorFlow • Made with ❤️ for Marine Conservation</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
