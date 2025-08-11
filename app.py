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
import logging

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

# Google Drive file IDs (you need to update these with your actual file IDs)
GOOGLE_DRIVE_FILES = {
    'cnn_model': {
        'id': '1qsVB_P2GIOdzwuzKHtazmTPafE7Nwb_P',   
        'filename': 'best_cnn_model.keras'
    },
    'rf_classifier': {
        'id': '1hQ8IkUWZJjVKd-rtKaLriBa9kQVscUeT',  
        'filename': 'best_classifier.joblib'
    }
}

def check_tensorflow_version():
    """Check TensorFlow version and show compatibility info"""
    tf_version = tf.__version__
    st.sidebar.write(f"TensorFlow Version: {tf_version}")
    
    # Check if version is compatible
    major, minor = tf_version.split('.')[:2]
    major, minor = int(major), int(minor)
    
    if major >= 2 and minor >= 15:
        st.sidebar.success("✅ TensorFlow version compatible")
        return True
    else:
        st.sidebar.warning("⚠️ TensorFlow version may cause compatibility issues")
        return False

def download_file_from_gdrive(file_id, filename):
    """Download file from Google Drive using gdown with error handling"""
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
        
        try:
            gdown.download(url, str(file_path), quiet=False)
        except Exception as download_error:
            st.error(f"Download failed with gdown: {download_error}")
            
            # Try alternative download method
            st.info("Trying alternative download method...")
            response = requests.get(url)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            else:
                raise Exception(f"Failed to download with status code: {response.status_code}")
        
        if file_path.exists():
            st.success(f"✅ {filename} downloaded successfully!")
            return str(file_path)
        else:
            st.error(f"❌ Failed to download {filename}")
            return None
            
    except Exception as e:
        st.error(f"❌ Error downloading {filename}: {str(e)}")
        return None

def load_model_with_compatibility(model_path):
    """Load model with various compatibility methods"""
    
    # Method 1: Standard loading
    try:
        st.info("Attempting standard model loading...")
        model = load_model(model_path)
        st.success("✅ Model loaded successfully with standard method")
        return model
    except Exception as e:
        st.warning(f"Standard loading failed: {str(e)[:100]}...")
    
    # Method 2: Load with compile=False
    try:
        st.info("Attempting to load model without compilation...")
        model = load_model(model_path, compile=False)
        st.success("✅ Model loaded successfully without compilation")
        return model
    except Exception as e:
        st.warning(f"Loading without compilation failed: {str(e)[:100]}...")
    
    # Method 3: Try with custom objects
    try:
        st.info("Attempting to load with custom objects...")
        custom_objects = {
            'Functional': tf.keras.Model,
            'functional': tf.keras.Model,
        }
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        st.success("✅ Model loaded successfully with custom objects")
        return model
    except Exception as e:
        st.warning(f"Loading with custom objects failed: {str(e)[:100]}...")
    
    # Method 4: Try loading weights only (requires model architecture)
    try:
        st.info("Attempting to recreate model architecture and load weights...")
        # Create a simple VGG16-based model architecture
        model = create_vgg16_model_architecture()
        
        # Try to load weights
        try:
            model.load_weights(model_path.replace('.keras', '_weights.h5'))
        except:
            # If weights file doesn't exist, extract from .keras file
            temp_model = tf.keras.models.load_model(model_path, compile=False)
            model.set_weights(temp_model.get_weights())
        
        st.success("✅ Model recreated and weights loaded successfully")
        return model
    except Exception as e:
        st.warning(f"Weight loading failed: {str(e)[:100]}...")
    
    return None

def create_vgg16_model_architecture():
    """Recreate the VGG16 model architecture manually"""
    try:
        # Create VGG16 base
        base_model = tf.keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze base layers except last few
        for layer in base_model.layers[:-4]:
            layer.trainable = False
        
        # Add custom top layers
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        # Feature layer
        features = tf.keras.layers.Dense(256, activation='relu', name='feature_layer')(x)
        features = tf.keras.layers.Dropout(0.4)(features)
        
        # Output layer
        outputs = tf.keras.layers.Dense(2, activation='softmax', name='predictions')(features)
        
        model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
        
        return model
        
    except Exception as e:
        st.error(f"Failed to create model architecture: {e}")
        return None

@st.cache_resource
def load_models():
    """Load the trained CNN and Random Forest models with error handling"""
    
    # Check TensorFlow version
    check_tensorflow_version()
    
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
        
        # Load CNN model with compatibility methods
        with st.spinner("Loading CNN model..."):
            cnn_model = load_model_with_compatibility(cnn_path)
        
        if cnn_model is None:
            st.error("❌ Failed to load CNN model with all methods")
            return None, None, None
        
        # Create feature extractor
        try:
            # Try to get feature layer by name first
            feature_layer = None
            for layer in cnn_model.layers:
                if 'feature' in layer.name.lower():
                    feature_layer = layer
                    break
            
            if feature_layer:
                feature_extractor = tf.keras.models.Model(
                    inputs=cnn_model.inputs, 
                    outputs=feature_layer.output
                )
                st.info(f"✅ Feature extractor created using layer: {feature_layer.name}")
            else:
                # Use second to last layer as fallback
                feature_extractor = tf.keras.models.Model(
                    inputs=cnn_model.inputs, 
                    outputs=cnn_model.layers[-2].output
                )
                st.info("✅ Feature extractor created using second-to-last layer")
                
        except Exception as e:
            st.error(f"❌ Error creating feature extractor: {str(e)}")
            return None, None, None
        
        # Load Random Forest classifier
        with st.spinner("Loading Random Forest classifier..."):
            try:
                rf_classifier = joblib.load(rf_path)
                st.success("✅ Random Forest classifier loaded successfully")
            except Exception as e:
                st.error(f"❌ Error loading Random Forest classifier: {str(e)}")
                return None, None, None
        
        return cnn_model, feature_extractor, rf_classifier
        
    except Exception as e:
        st.error(f"❌ Error in load_models: {str(e)}")
        st.error("Please check the Google Drive file IDs and permissions.")
        return None, None, None

def preprocess_image(image):
    """Preprocess image exactly as done during training"""
    try:
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
        
    except Exception as e:
        st.error(f"Error in image preprocessing: {str(e)}")
        return None

def predict_anemia(image, feature_extractor, rf_classifier):
    """Make prediction using the CNN+RF pipeline"""
    try:
        # Preprocess the image
        processed_img = preprocess_image(image)
        if processed_img is None:
            return None, None, None
        
        # Extract features using CNN
        features = feature_extractor.predict(processed_img, verbose=0)
        
        # Make prediction using Random Forest
        prediction = rf_classifier.predict(features)[0]
        prediction_proba = rf_classifier.predict_proba(features)[0]
        
        return prediction, prediction_proba, features
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

def main():
    st.title("🩺 Anemia Detection from Conjunctiva Images")
    st.markdown("---")
    
    # Display important note about first run
    st.info("📋 **First Time Setup**: Models will be downloaded from Google Drive on first run. This may take a few minutes.")
    
    # Add troubleshooting section
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        **Common Issues and Solutions:**
        
        1. **Model Loading Errors**: 
           - Usually caused by TensorFlow version mismatches
           - The app tries multiple loading methods automatically
           - Check TensorFlow version in sidebar
        
        2. **Download Issues**:
           - Ensure Google Drive files are publicly accessible
           - Check file IDs are correct
           - Verify internet connection
        
        3. **Prediction Errors**:
           - Ensure image is clear and well-lit
           - Try different image formats (JPG, PNG)
           - Check image is not corrupted
        """)
    
    # Load models
    cnn_model, feature_extractor, rf_classifier = load_models()
    
    if cnn_model is None or feature_extractor is None or rf_classifier is None:
        st.error("❌ Unable to load models. Please check the troubleshooting section above.")
        st.info("**Possible solutions:**")
        st.write("1. Check Google Drive file IDs and permissions")
        st.write("2. Ensure files are publicly accessible")
        st.write("3. Try refreshing the page")
        st.write("4. Check TensorFlow version compatibility")
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
                        result = predict_anemia(image, feature_extractor, rf_classifier)
                        
                        if result[0] is not None:
                            prediction, prediction_proba, features = result
                            
                            # Store results in session state for display in col2
                            st.session_state.prediction_results = {
                                'prediction': prediction,
                                'prediction_proba': prediction_proba,
                                'features': features,
                                'image': image
                            }
                        else:
                            st.error("❌ Prediction failed. Please try with a different image.")
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        logging.error(f"Prediction error: {str(e)}")
    
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
            try:
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
                plt.close(fig)  # Prevent memory leaks
                
            except Exception as e:
                st.error(f"Error creating probability chart: {str(e)}")
            
            # Feature visualization
            st.subheader("🧠 Extracted Features")
            st.write(f"The CNN extracted {results['features'].shape[1]} features from the image.")
            
            # Show feature distribution
            try:
                fig2, ax2 = plt.subplots(figsize=(10, 3))
                ax2.hist(results['features'][0], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax2.set_xlabel('Feature Values')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Distribution of Extracted Features')
                st.pyplot(fig2)
                plt.close(fig2)  # Prevent memory leaks
                
            except Exception as e:
                st.error(f"Error creating feature visualization: {str(e)}")
            
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
    - VGG16 Feature Extractor (224x224 input)
    - Random Forest Classifier (256 features)
    
    **Training Dataset:**
    - 1,268 conjunctiva images
    - Binary classification (Anemic/Non-anemic)
    
    **Performance Metrics:**
    - Accuracy: 93.19%
    - Precision: 93.20%
    - Recall: 93.19%
    - F1-score: 93.19%
    - AUC: 97.45%
    """)
    
    # System information
    st.sidebar.header("🖥️ System Info")
    check_tensorflow_version()
    st.sidebar.write(f"Python Version: {st.version.python_version}")
    
    # Model information expander
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        ### Model Pipeline:
        1. **Image Preprocessing**: Resize to 224×224, normalize to [0,1]
        2. **Feature Extraction**: VGG16 CNN extracts 256-dimensional feature vectors
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
        
        ### Compatibility Features:
        - Multiple model loading methods for different TensorFlow versions
        - Automatic fallback to alternative loading strategies
        - Error handling and troubleshooting guidance
        - Version compatibility checking
        """)

if __name__ == "__main__":
    main()