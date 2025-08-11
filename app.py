import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
from pathlib import Path
import gdown

# Configure Streamlit page
st.set_page_config(
    page_title="Anemia Detection from Conjunctiva Images",
    page_icon="🩺",
    layout="wide"
)

# Constants matching your training setup
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ['Anemic', 'Non-anemic']

# Google Drive file IDs (extract from your drive links)
GOOGLE_DRIVE_FILES = {
    'cnn_model': {
        'id': '1qsVB_P2GIOdzwuzKHtazmTPafE7Nwb_P',   # Extract this from your drive link
        'filename': 'best_cnn_model.keras'
    },
    'rf_classifier': {
        'id': '1hQ8IkUWZJjVKd-rtKaLriBa9kQVscUeT',  # You need to upload RF classifier to drive and get this ID
        'filename': 'best_classifier.joblib'
    }
}

def download_file_from_gdrive(file_id, filename):
    """Download file from Google Drive using gdown"""
    try:
        # Create models directory if it doesn't exist
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        file_path = models_dir / filename
        
        # Skip download if file already exists
        if file_path.exists():
            st.info(f"✅ {filename} already exists, skipping download.")
            return str(file_path)
        
        # Download from Google Drive
        url = f"https://drive.google.com/uc?id={file_id}"
        st.info(f"📥 Downloading {filename}...")
        
        gdown.download(url, str(file_path), quiet=False)
        
        if file_path.exists():
            st.success(f"✅ {filename} downloaded successfully!")
            return str(file_path)
        else:
            st.error(f"❌ Failed to download {filename}")
            return None
            
    except Exception as e:
        st.error(f"❌ Error downloading {filename}: {str(e)}")
        return None

@st.cache_resource
def load_models():
    """Load the trained CNN and Random Forest models"""
    try:
        # Download models from Google Drive
        with st.spinner("Downloading models from Google Drive..."):
            cnn_path = download_file_from_gdrive(
                GOOGLE_DRIVE_FILES['cnn_model']['id'],
                GOOGLE_DRIVE_FILES['cnn_model']['filename']
            )
            
            rf_path = download_file_from_gdrive(
                GOOGLE_DRIVE_FILES['rf_classifier']['id'],
                GOOGLE_DRIVE_FILES['rf_classifier']['filename']
            )
        
        if not cnn_path or not rf_path:
            st.error("❌ Failed to download required model files")
            return None, None, None
        
        # Load CNN model
        with st.spinner("Loading CNN model..."):
            cnn_model = load_model(cnn_path)
        
        # Create feature extractor (second to last layer)
        feature_extractor = tf.keras.models.Model(
            inputs=cnn_model.inputs, 
            outputs=cnn_model.layers[-2].output
        )
        
        # Load Random Forest classifier
        with st.spinner("Loading Random Forest classifier..."):
            rf_classifier = joblib.load(rf_path)
        
        return cnn_model, feature_extractor, rf_classifier
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.error("Please check the Google Drive file IDs and permissions.")
        return None, None, None

def preprocess_image(image):
    """Preprocess image exactly as done during training"""
    # Convert PIL image to OpenCV format
    image_array = np.array(image)
    
    # Convert RGB to BGR if necessary (OpenCV uses BGR)
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # Resize image to match training dimensions
    resized_img = cv2.resize(image_array, (IMG_WIDTH, IMG_HEIGHT))
    
    # Normalize pixel values to [0, 1]
    normalized_img = resized_img / 255.0
    
    # Add batch dimension
    processed_img = np.expand_dims(normalized_img, axis=0)
    
    return processed_img

def predict_anemia(image, feature_extractor, rf_classifier):
    """Make prediction using the CNN+RF pipeline"""
    # Preprocess the image
    processed_img = preprocess_image(image)
    
    # Extract features using CNN
    features = feature_extractor.predict(processed_img, verbose=0)
    
    # Make prediction using Random Forest
    prediction = rf_classifier.predict(features)[0]
    prediction_proba = rf_classifier.predict_proba(features)[0]
    
    return prediction, prediction_proba, features

def main():
    st.title("🩺 Anemia Detection from Conjunctiva Images")
    st.markdown("---")
    
    # Display important note about first run
    st.info("📋 **First Time Setup**: Models will be downloaded from Google Drive on first run. This may take a few minutes.")
    
    # Load models
    cnn_model, feature_extractor, rf_classifier = load_models()
    
    if cnn_model is None or feature_extractor is None or rf_classifier is None:
        st.error("❌ Unable to load models. Please check the setup and try again.")
        st.stop()
    
    st.success("✅ Models loaded successfully!")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a conjunctiva image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a clear image of the conjunctiva (inner eyelid)"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add prediction button
            if st.button("🔍 Analyze for Anemia", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Make prediction
                        prediction, prediction_proba, features = predict_anemia(
                            image, feature_extractor, rf_classifier
                        )
                        
                        # Store results in session state for display in col2
                        st.session_state.prediction_results = {
                            'prediction': prediction,
                            'prediction_proba': prediction_proba,
                            'features': features,
                            'image': image
                        }
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
    
    with col2:
        st.header("📊 Analysis Results")
        
        if 'prediction_results' in st.session_state:
            results = st.session_state.prediction_results
            prediction = results['prediction']
            prediction_proba = results['prediction_proba']
            
            # Display prediction
            predicted_class = CLASS_NAMES[prediction]
            confidence = prediction_proba[prediction] * 100
            
            # Color code the result
            if prediction == 0:  # Anemic
                st.error(f"🚨 **Prediction: {predicted_class}**")
            else:  # Non-anemic
                st.success(f"✅ **Prediction: {predicted_class}**")
            
            st.metric("Confidence", f"{confidence:.2f}%")
            
            # Display probability distribution
            st.subheader("📈 Prediction Probabilities")
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(CLASS_NAMES, prediction_proba, 
                         color=['#ff6b6b' if i == 0 else '#51cf66' for i in range(len(CLASS_NAMES))])
            ax.set_ylabel('Probability')
            ax.set_title('Classification Probabilities')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, prob in zip(bars, prediction_proba):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{prob:.3f}', ha='center', va='bottom')
            
            st.pyplot(fig)
            
            # Feature visualization
            st.subheader("🧠 Extracted Features")
            st.write(f"The CNN extracted {results['features'].shape[1]} features from the image.")
            
            # Show feature distribution
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ax2.hist(results['features'][0], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_xlabel('Feature Values')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Extracted Features')
            st.pyplot(fig2)
            
            # Medical disclaimer
            st.warning("""
            ⚠️ **Medical Disclaimer**: 
            This is an AI-based screening tool and should not be used as a substitute for professional medical diagnosis. 
            Please consult with a healthcare professional for proper medical evaluation and treatment.
            """)
        
        else:
            st.info("👆 Upload an image and click 'Analyze for Anemia' to see results here.")
    
    # Sidebar with information
    st.sidebar.header("ℹ️ About This App")
    st.sidebar.markdown("""
    This application uses a hybrid CNN+Random Forest model to detect anemia from conjunctiva images.
    
    **Model Architecture:**
    - CNN Feature Extractor (224x224 input)
    - Random Forest Classifier (128 features)
    
    **Training Dataset:**
    - 1,268 conjunctiva images
    - Binary classification (Anemic/Non-anemic)
    
    **Performance Metrics:**
    - Accuracy: 96.85%
    - Precision: 96.89%
    - Recall: 96.85%
    - F1-score: 96.85%
    """)
    
    # Model information expander
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        ### Model Pipeline:
        1. **Image Preprocessing**: Resize to 224×224, normalize to [0,1]
        2. **Feature Extraction**: CNN extracts 128-dimensional feature vectors
        3. **Classification**: Random Forest makes final prediction
        
        ### Image Requirements:
        - Clear, well-lit conjunctiva images
        - Supported formats: JPG, JPEG, PNG, BMP
        - Optimal resolution: At least 224×224 pixels
        
        ### How to Use:
        1. Upload a conjunctiva image using the file uploader
        2. Click "Analyze for Anemia" to get predictions
        3. Review the results and confidence scores
        4. Always consult a medical professional for diagnosis
        
        ### First Time Setup:
        - Models are automatically downloaded from Google Drive
        - Initial setup may take a few minutes
        - Subsequent runs will be faster as models are cached
        """)

if __name__ == "__main__":
    main()