import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

def load_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def extract_features(image, model):
    # Resize for ResNet50
    img = cv2.resize(image, (224, 224))

    # Convert to array
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    # ResNet50 preprocessing
    img = preprocess_input(img)

    # Extract features
    features = model.predict(img, verbose=0)
    return features
