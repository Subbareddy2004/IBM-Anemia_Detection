# 🩺 Anemia Detection Using Conjunctiva Images

A deep learning application that detects anemia from conjunctiva (inner eyelid) images using a hybrid CNN + Random Forest model architecture.

## 🚀 Live Demo

**[Try the Live App Here](YOUR_STREAMLIT_APP_URL_HERE)** *(Replace with your actual Streamlit Cloud URL)*

## 📊 Model Performance

- **Accuracy**: 96.85%
- **Precision**: 96.89% 
- **Recall**: 96.85%
- **F1-Score**: 96.85%
- **Dataset Size**: 1,268 conjunctiva images

## 🏗️ Model Architecture

### Hybrid CNN + Random Forest Pipeline:
1. **CNN Feature Extractor**: Extracts 128-dimensional feature vectors from 224×224 conjunctiva images
2. **Random Forest Classifier**: Makes final binary classification (Anemic/Non-anemic)

### CNN Architecture:
```
Sequential Model:
├── Conv2D (32 filters, 3×3) + ReLU + MaxPool2D
├── Conv2D (64 filters, 3×3) + ReLU + MaxPool2D  
├── Conv2D (128 filters, 3×3) + ReLU + MaxPool2D
├── Flatten
├── Dense (128 units, ReLU) ← Feature Extraction Layer
└── Dense (2 units, Softmax) ← Classification Layer
```

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow/Keras
- **Machine Learning**: Scikit-learn (Random Forest)
- **Web Framework**: Streamlit
- **Image Processing**: OpenCV, PIL
- **Data Visualization**: Matplotlib, Seaborn
- **Model Storage**: Google Drive (for large model files)
- **Deployment**: Streamlit Cloud

## 📁 Project Structure

```
├── Production/
│   ├── app.py              # Main Streamlit application
│   ├── requirements.txt    # Python dependencies
│   └── README.md          # This file
├── Training_Notebooks/     # Jupyter notebooks used for model training
├── Models/                 # Local model files (not pushed to GitHub)
│   ├── best_cnn_model.keras
│   └── best_classifier.joblib
└── assets/                 # Images and documentation assets
```

## 🚀 Quick Start

### Option 1: Use the Live Web App
Simply visit the [live demo](YOUR_STREAMLIT_APP_URL_HERE) and upload a conjunctiva image to get instant results.

### Option 2: Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/anemia-detection.git
   cd anemia-detection/Production
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

## 📋 Usage Instructions

1. **Upload Image**: Choose a clear conjunctiva (inner eyelid) image
2. **Analyze**: Click "Analyze for Anemia" button
3. **View Results**: 
   - Prediction (Anemic/Non-anemic)
   - Confidence score
   - Probability distribution
   - Feature visualization

### 📸 Image Requirements
- **Format**: JPG, JPEG, PNG, BMP
- **Quality**: Clear, well-lit conjunctiva images
- **Resolution**: Minimum 224×224 pixels recommended
- **Subject**: Inner eyelid (conjunctiva) area

## 🧠 How It Works

1. **Image Preprocessing**: 
   - Resize to 224×224 pixels
   - Normalize pixel values to [0,1]
   - Convert to appropriate color format

2. **Feature Extraction**: 
   - CNN processes the image through convolutional layers
   - Extracts 128-dimensional feature vector from dense layer

3. **Classification**: 
   - Random Forest classifier uses extracted features
   - Outputs binary prediction with confidence scores

## 📈 Training Details

- **Dataset**: 1,268 conjunctiva images (Anemic vs Non-anemic)
- **Data Split**: 80% training, 20% validation
- **Training Epochs**: 10
- **Batch Size**: 32
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Data Augmentation**: Rotation, shifting, flipping, brightness adjustment

## 🔧 Technical Implementation

### Model Storage Strategy
- **Challenge**: Large model files (>100MB) cannot be stored on GitHub
- **Solution**: Models hosted on Google Drive with automatic download
- **Benefits**: 
  - First-time setup downloads models automatically
  - Subsequent runs use cached models for faster loading
  - No manual model file management required

### Cloud Deployment
- **Platform**: Streamlit Cloud
- **Features**:
  - Automatic model downloading on first run
  - Responsive web interface
  - Real-time predictions
  - Interactive visualizations

## ⚠️ Medical Disclaimer

**IMPORTANT**: This application is for educational and research purposes only. It should **NOT** be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical evaluation and treatment of anemia or any other health conditions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Original CNN+RF architecture inspiration from research in medical image analysis
- Training dataset contributors and medical professionals who validated the approach
- Open-source community for the excellent tools and libraries used in this project

## 📊 Performance Metrics Details

| Metric | Value | Description |
|--------|--------|-------------|
| Accuracy | 96.85% | Overall correct predictions |
| Precision | 96.89% | True positives / (True positives + False positives) |
| Recall | 96.85% | True positives / (True positives + False negatives) |
| F1-Score | 96.85% | Harmonic mean of precision and recall |

## 🔍 Future Enhancements

- [ ] Support for batch image processing
- [ ] Integration with additional biomarkers
- [ ] Mobile app development
- [ ] Multi-language support
- [ ] Enhanced visualization dashboard
- [ ] Real-time camera integration

---

**⭐ If you found this project helpful, please give it a star!**