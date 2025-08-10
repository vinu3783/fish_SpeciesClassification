🐟 Fish Species Classification Project
An end-to-end deep learning project for classifying fish species from images using CNN and Transfer Learning techniques. Built with TensorFlow/Keras and deployed as a Streamlit web application.
🎯 Project Overview
This project implements a complete machine learning pipeline for fish species classification:

Data Preprocessing: Automated image preprocessing with smart augmentation
Model Training: CNN from scratch + 5 pre-trained models (VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0)
Model Evaluation: Comprehensive performance comparison and metrics
Web Deployment: Interactive Streamlit application for real-time predictions
🏆 Key Features
Multi-Model Architecture: Compare performance across different model architectures
Smart Data Augmentation: Automatically adjusts augmentation based on dataset size and class balance
Real-time Web App: Upload images and get instant AI-powered predictions
Comprehensive Evaluation: Detailed metrics, confusion matrices, and performance visualizations
Model Comparison: Side-by-side comparison of all trained models
Export Results: Download predictions and analysis reports
📋 Business Use Cases
Fisheries Management: Automated species identification for catch monitoring
Marine Research: Rapid classification for ecological studies
Commercial Applications: Quality control in seafood processing
Educational Tools: Interactive learning applications for marine biology
🚀 Quick Start
Prerequisites
Python 3.8+
pip (Python package manager)
Installation
Clone the repository:
git clone https://github.com/YOUR_USERNAME/fish-classification.git
cd fish-classification
Install dependencies:
pip install -r requirements.txt
Setup project structure:
python setup_environment.py
Dataset Setup
Prepare your fish dataset with the following structure:
data/
├── train/
│   ├── species1/
│   ├── species2/
│   └── ...
├── test/
│   ├── species1/
│   ├── species2/
│   └── ...
└── val/
    ├── species1/
    ├── species2/
    └── ...
    Place images in JPG/PNG format in respective species folders
Training Models
Run the complete training pipeline:

# Execute the combined preprocessing + training pipeline
python combined_training_pipeline.py
This will:

✅ Preprocess your dataset
✅ Train multiple models (CNN + Transfer Learning)
✅ Compare performance metrics
✅ Save best models automatically
Launch Web App
cd streamlit_app
streamlit run app.py
Visit http://localhost:8501 to use the web application!

📁 Project Structure
fish-classification/
├── 📂 data/                          # Dataset (not included in repo)
│   ├── train/
│   ├── test/
│   └── val/
├── 📂 src/                           # Source code
│   ├── data_preprocessing/
│   ├── model_training/
│   ├── evaluation/
│   └── deployment/
├── 📂 streamlit_app/                 # Web application
│   ├── app.py                        # Full-featured app
│   ├── simple_fish_app.py           # Simplified app
│   └── run_streamlit.py             # Launch script
├── 📂 results/                       # Training results (not in repo)
│   ├── models/                       # Saved models (.h5 files)
│   ├── plots/                        # Visualization plots
│   └── reports/                      # Performance reports
├── 📄 combined_training_pipeline.py  # Main training script
├── 📄 setup_environment.py          # Environment setup
├── 📄 config.py                     # Configuration
├── 📄 utils.py                      # Utility functions
├── 📄 requirements.txt              # Dependencies
└── 📄 README.md                     # This file
🤖 Model Architectures
The project implements and compares multiple deep learning architectures:

1. Custom CNN (From Scratch)
Architecture: 3 Conv2D blocks + Global Average Pooling + Dense layers
Parameters: ~1.2M parameters
Use Case: Baseline comparison and custom feature learning
📊 Performance Metrics
The project evaluates models using:

Accuracy: Overall classification accuracy
Top-3 Accuracy: Correct class in top 3 predictions
Precision/Recall: Per-class performance metrics
F1-Score: Harmonic mean of precision and recall
Confusion Matrix: Detailed classification breakdown
Training Time: Model efficiency comparison
🎨 Data Augmentation Strategy
Smart augmentation based on dataset characteristics:

Light (>5K images): Horizontal flip + rotation
Moderate (2K-5K images): + brightness + zoom + shifts
Heavy (<2K images): + shear + advanced transformations
🔧 Configuration
Key parameters in config.py:

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
📱 Streamlit Web Application
Features:
Image Upload: Drag & drop or browse for fish images
Real-time Prediction: Instant AI classification
Confidence Visualization: Interactive charts showing prediction confidence
Model Comparison: Switch between different trained models
Image Enhancement: Adjust brightness, contrast, sharpness
Export Results: Download predictions as JSON
Usage:
Upload a fish image (PNG/JPG)
Select model for prediction
Click "Classify Fish Species"
View results and confidence scores
Download results if needed
🛠️ Development
Adding New Models
Add model creation function in src/model_training/
Update model list in training pipeline
Test and evaluate performance
Update documentation
Custom Dataset
Organize images in train/test/val structure
Update class names in config
Run preprocessing pipeline
Train models and evaluate
📈 Results Analysis
The training pipeline generates:

Performance comparison charts
Training history plots
Confusion matrices
Model comparison table
Detailed JSON reports
🚀 Deployment Options
Local Deployment
streamlit run streamlit_app/app.py
Cloud Deployment
Streamlit Cloud: Push to GitHub and deploy
Heroku: Use provided Dockerfile
AWS/GCP: Deploy as containerized application
🤝 Contributing
Fork the repository
Create feature branch: git checkout -b feature/new-feature
Commit changes: git commit -am 'Add new feature'
Push to branch: git push origin feature/new-feature
Create Pull Request
📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
TensorFlow/Keras for deep learning framework
Streamlit for web application framework
Pre-trained models from TensorFlow Hub
Fish datasets from various marine biology sources
📞 Contact
GitHub:https://github.com/vinu3783
Email:vinayakagc210@gmail.com
LinkedIn: Your linkedin.com/in/vinayaka-gc-54817a259
