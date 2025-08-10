# Multiclass Fish Image Classification

## Project Overview
This project implements a deep learning solution for classifying fish images into multiple species categories using both custom CNN and transfer learning approaches.

## Features
- Custom CNN model training from scratch
- Transfer learning with 5 pre-trained models (VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0)
- Data augmentation for improved model robustness
- Comprehensive model evaluation and comparison
- Interactive Streamlit web application for real-time predictions
- Model performance visualization and reporting

## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd fish_classification_project
```

### 2. Create virtual environment
```bash
python -m venv fish_env
source fish_env/bin/activate  # On Windows: fish_env\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Test setup
```bash
python test_setup.py
```

### 5. Add your dataset
Place your fish dataset in the `data/` folder following this structure:
```
data/
├── train/
│   ├── species1/
│   ├── species2/
│   └── ...
├── test/
└── val/
```

## Usage

### Training Models
```bash
python src/train_models.py
```

### Running Streamlit App
```bash
streamlit run streamlit_app/app.py
```

### Model Evaluation
```bash
python src/evaluate_models.py
```

## Project Structure
- `src/` - Core training and evaluation scripts
- `streamlit_app/` - Web application files
- `models/` - Saved trained models
- `results/` - Training results and visualizations
- `notebooks/` - Jupyter notebooks for experimentation

## Technologies Used
- **Deep Learning**: TensorFlow, Keras
- **Data Processing**: NumPy, Pandas, OpenCV
- **Visualization**: Matplotlib, Seaborn
- **Web App**: Streamlit
- **Image Processing**: PIL, Albumentations

## Results
Model performance metrics and comparisons will be available in the `results/` folder after training.

## Contributing
Feel free to contribute to this project by submitting pull requests or reporting issues.

## License
This project is licensed under the MIT License.