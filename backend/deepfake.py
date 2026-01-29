# deepfake_detection_system.py

# Import necessary libraries
import os
import cv2
import dlib
import numpy as np
import joblib
import pandas as pd
import urllib.request
import bz2
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50  # You can switch to MobileNetV2 if desired
from tensorflow.keras.layers import Dense, Flatten, TimeDistributed, LSTM
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from flask import Flask, request, jsonify
from flask_cors import CORS

# Enable memory growth for GPUs (if using a GPU)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPUs")
    except RuntimeError as e:
        print(e)

# Suppress TensorFlow warnings about memory allocation
import logging
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Paths to data directories and models
DATA_DIR = 'D:/DARSHAN/Project/Deep Fake Video Detection/deepfake_detection_systemv2/deepfake_detection_systemv2/archive'  # Directory containing 'real_videos/' and 'fake_videos/'
SPATIAL_MODEL_PATH = 'models/spatial_model.h5'
TEMPORAL_MODEL_PATH = 'models/temporal_model.h5'
FREQUENCY_MODEL_PATH = 'models/frequency_model.pkl'
BIOMETRIC_MODEL_PATH = 'models/biometric_model.pkl'

# Global variables
SEQUENCE_LENGTH = 10
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8  # Reduced batch size to lower memory usage

def download_shape_predictor():
    url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    compressed_file = 'shape_predictor_68_face_landmarks.dat.bz2'
    dat_file = 'shape_predictor_68_face_landmarks.dat'

    if not os.path.exists(dat_file):
        print("Downloading shape_predictor_68_face_landmarks.dat.bz2...")
        urllib.request.urlretrieve(url, compressed_file)
        print("Download completed.")

        print("Extracting shape_predictor_68_face_landmarks.dat...")
        with bz2.BZ2File(compressed_file) as fr, open(dat_file, 'wb') as fw:
            fw.write(fr.read())
        print("Extraction completed.")

        # Remove the compressed file to save space
        os.remove(compressed_file)
    else:
        print("shape_predictor_68_face_landmarks.dat already exists.")

download_shape_predictor()

# Initialize Dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()

predictor_path = 'shape_predictor_68_face_landmarks.dat'  # File is in the current directory

if not os.path.exists(predictor_path):
    print(f"Error: {predictor_path} not found.")
else:
    predictor = dlib.shape_predictor(predictor_path)
    print("Shape predictor loaded successfully.")

# Create necessary directories
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('preprocessed_data'):
    os.makedirs('preprocessed_data')

# ===========================
# Data Preprocessing
# ===========================

def preprocess_data():
    """
    Preprocess videos by extracting frames and organizing them into directories.
    """
    preprocessed_dir = 'preprocessed_data'
    classes = ['real_videos', 'fake_videos']
    for cls in classes:
        video_dir = os.path.join(DATA_DIR, cls)
        frame_output_dir = os.path.join(preprocessed_dir, cls)
        print(frame_output_dir)
        if not os.path.exists(frame_output_dir):
            os.makedirs(frame_output_dir)

        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        for video_name in video_files:
            video_path = os.path.join(video_dir, video_name)
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            success, frame = cap.read()
            while success:
                frame_file = f"{os.path.splitext(video_name)[0]}frame{frame_count}.jpg"
                frame_path = os.path.join(frame_output_dir, frame_file)
                cv2.imwrite(frame_path, frame)
                success, frame = cap.read()
                frame_count += 1
            cap.release()
            print(f"Processed video: {video_name}")

# Function to create a DataFrame with limited samples
def create_dataframe(data_dir, max_samples_per_class=1000):
    """
    Create a pandas DataFrame containing file paths and labels for a subset of data.
    """
    data = []
    classes = ['real_videos', 'fake_videos']
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        frame_names = os.listdir(cls_dir)[:max_samples_per_class]
        for frame_name in frame_names:
            frame_path = os.path.join(cls_dir, frame_name)
            print(frame_path)
            label = cls
            data.append({'filename': frame_path, 'class': label})
    df = pd.DataFrame(data)
    return df

# ===========================
# Training the Spatial Analysis Model
# ===========================

def train_spatial_model():
    # Optionally, switch to a lighter base model like MobileNetV2 to reduce memory usage
    # from tensorflow.keras.applications import MobileNetV2
    # base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
    # For this example, we'll continue using ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
    for layer in base_model.layers:
        layer.trainable = False  # Freeze the base model layers

    # Add custom layers
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)  # Reduced the number of neurons to lower memory usage
    predictions = Dense(1, activation='sigmoid')(x)

    # Build the model
    spatial_model = Model(inputs=base_model.input, outputs=predictions)
    spatial_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Create dataframes for training and validation with limited samples
    train_df = create_dataframe('preprocessed_data/', max_samples_per_class=1000)
    validation_df = create_dataframe('preprocessed_data/', max_samples_per_class=250)

    # Data generators using flow_from_dataframe
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filename',
        y_col='class',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    validation_generator = train_datagen.flow_from_dataframe(
        validation_df,
        x_col='filename',
        y_col='class',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    # Train the model
    spatial_model.fit(
        train_generator,
        epochs=5,  # Adjust the number of epochs as needed
        validation_data=validation_generator
    )

    # Save the model
    spatial_model.save(SPATIAL_MODEL_PATH)
    print("Spatial model trained and saved.")

# ===========================
# Training the Temporal Analysis Model
# ===========================

def create_sequence_generator(directory, batch_size, sequence_length, max_videos_per_class=50):
    """
    Custom generator to yield sequences of frames.
    """
    class_map = {'real_videos': 0, 'fake_videos': 1}
    classes = ['real_videos', 'fake_videos']
    while True:
        sequences = []
        labels = []
        for cls in classes:
            cls_dir = os.path.join(directory, cls)
            video_files = [f for f in os.listdir(cls_dir) if f.endswith('.mp4')][:max_videos_per_class]
            for video_name in video_files:
                video_path = os.path.join(cls_dir, video_name)
                cap = cv2.VideoCapture(video_path)
                frames = []
                success, frame = cap.read()
                while success and len(frames) < sequence_length:
                    frame = cv2.resize(frame, IMAGE_SIZE)
                    frame = frame.astype('float32') / 255.0
                    frames.append(frame)
                    success, frame = cap.read()
                cap.release()
                if len(frames) == sequence_length:
                    sequences.append(frames)
                    labels.append(class_map[cls])
                if len(sequences) == batch_size:
                    yield np.array(sequences), np.array(labels)
                    sequences = []
                    labels = []

def train_temporal_model():
    # Optionally, switch to a lighter base model like MobileNetV2
    # from tensorflow.keras.applications import MobileNetV2
    # base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
    for layer in base_model.layers:
        layer.trainable = False

    # Build the LSTM model
    temporal_model = Sequential()
    temporal_model.add(TimeDistributed(base_model, input_shape=(SEQUENCE_LENGTH, *IMAGE_SIZE, 3)))
    temporal_model.add(TimeDistributed(Flatten()))
    temporal_model.add(LSTM(128, activation='tanh'))  # Reduced units to lower memory usage
    temporal_model.add(Dense(1, activation='sigmoid'))

    temporal_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Generators with limited number of videos
    train_generator = create_sequence_generator(DATA_DIR, BATCH_SIZE, SEQUENCE_LENGTH, max_videos_per_class=50)
    steps_per_epoch = 100  # Adjust based on your dataset
    validation_generator = create_sequence_generator(DATA_DIR, BATCH_SIZE, SEQUENCE_LENGTH, max_videos_per_class=10)
    validation_steps = 20  # Adjust based on your dataset

    # Train the model
    temporal_model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=5,  # Adjust the number of epochs
        validation_data=validation_generator,
        validation_steps=validation_steps
    )

    # Save the model
    temporal_model.save(TEMPORAL_MODEL_PATH)
    print("Temporal model trained and saved.")

# ===========================
# Training the Frequency Analysis Model
# ===========================

def train_frequency_model():
    X = []
    y = []
    class_map = {'real_videos': 0, 'fake_videos': 1}
    max_samples_per_class = 1000  # Limit the number of samples
    for cls in ['real_videos', 'fake_videos']:
        cls_dir = os.path.join('preprocessed_data/', cls)
        frame_names = os.listdir(cls_dir)[:max_samples_per_class]
        for frame_name in frame_names:
            frame_path = os.path.join(cls_dir, frame_name)
            frame = cv2.imread(frame_path)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dft = cv2.dft(np.float32(gray_frame), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
            features = magnitude_spectrum.flatten()[:5000]  # Limit to 5000 features
            X.append(features)
            y.append(class_map[cls])

    X = np.array(X)
    y = np.array(y)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train the SVM classifier
    frequency_model = SVC(kernel='linear', probability=True)
    frequency_model.fit(X_train, y_train)

    # Save the model
    joblib.dump(frequency_model, FREQUENCY_MODEL_PATH)
    print("Frequency model trained and saved.")

# ===========================
# Training the Biometric Analysis Model
# ===========================

def extract_biometric_features(max_videos_per_class=50):
    X = []
    y = []
    class_map = {'real_videos': 0, 'fake_videos': 1}
    for cls in ['real_videos', 'fake_videos']:
        video_dir = os.path.join(DATA_DIR, cls)
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')][:max_videos_per_class]
        for video_name in video_files:
            video_path = os.path.join(video_dir, video_name)
            label = class_map[cls]
            cap = cv2.VideoCapture(video_path)
            blink_count = 0
            frame_count = 0
            success, frame = cap.read()
            while success:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                if len(faces) > 0:
                    landmarks = predictor(gray, faces[0])
                    ear = compute_ear(landmarks)
                    if ear < 0.21:
                        blink_count += 1
                frame_count += 1
                success, frame = cap.read()
            cap.release()
            total_time = frame_count / 30  # Assuming 30 FPS
            blink_rate = blink_count / total_time if total_time > 0 else 0
            X.append([blink_rate])
            y.append(label)
    return np.array(X), np.array(y)

def train_biometric_model():
    X, y = extract_biometric_features()

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train the Random Forest classifier
    biometric_model = RandomForestClassifier(n_estimators=100)
    biometric_model.fit(X_train, y_train)

    # Save the model
    joblib.dump(biometric_model, BIOMETRIC_MODEL_PATH)
    print("Biometric model trained and saved.")

def compute_ear(landmarks):
    # Coordinates for left and right eyes
    left_eye = [landmarks.part(i) for i in range(36, 42)]
    right_eye = [landmarks.part(i) for i in range(42, 48)]

    # Compute EAR for both eyes
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    # Average EAR
    ear = (left_ear + right_ear) / 2.0
    return ear

def eye_aspect_ratio(eye_points):
    # Vertical eye landmarks distances
    A = np.linalg.norm(np.array([eye_points[1].x, eye_points[1].y]) - np.array([eye_points[5].x, eye_points[5].y]))
    B = np.linalg.norm(np.array([eye_points[2].x, eye_points[2].y]) - np.array([eye_points[4].x, eye_points[4].y]))
    # Horizontal eye landmarks distance
    C = np.linalg.norm(np.array([eye_points[0].x, eye_points[0].y]) - np.array([eye_points[3].x, eye_points[3].y]))
    # Compute EAR
    ear = (A + B) / (2.0 * C)
    return ear

def analyze_video(video_path):
    # Load the trained models
    try:
        spatial_model = load_model(SPATIAL_MODEL_PATH)
    except Exception as e:
        print(f"Error loading spatial model: {e}")
        return None
    try:
        temporal_model = load_model(TEMPORAL_MODEL_PATH)
    except Exception as e:
        print(f"Error loading temporal model: {e}")
        return None
    try:
        frequency_model = joblib.load(FREQUENCY_MODEL_PATH)
    except Exception as e:
        print(f"Error loading frequency model: {e}")
        return None
    try:
        biometric_model = joblib.load(BIOMETRIC_MODEL_PATH)
    except Exception as e:
        print(f"Error loading biometric model: {e}")
        return None

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    frames = []
    frame_count = 0
    success = True
    frame_skip = 5  # Process every 5th frame to reduce workload
    while success and frame_count < 500:  # Increased limit to 500 frames
        success, frame = cap.read()
        if success and frame_count % frame_skip == 0:
            if frame is not None:
                frames.append(frame)
        frame_count += 1
    cap.release()

    print(f"Number of frames extracted: {len(frames)}")
    if len(frames) < SEQUENCE_LENGTH:
        print("Not enough frames in the video for analysis.")
        return None

    # Preprocess frames
    try:
        frames_resized = [cv2.resize(f, IMAGE_SIZE) for f in frames]
        frames_normalized = [f.astype('float32') / 255.0 for f in frames_resized]
    except Exception as e:
        print(f"Error during frame preprocessing: {e}")
        return None

    # Spatial Analysis on the first frame
    try:
        spatial_frame = np.expand_dims(frames_normalized[0], axis=0)
        spatial_prediction = spatial_model.predict(spatial_frame)
        spatial_result = spatial_prediction[0][0]
    except Exception as e:
        print(f"Error during spatial analysis: {e}")
        return None

    # Temporal Analysis on a sequence of frames
    try:
        temporal_frames = frames_normalized[:SEQUENCE_LENGTH]
        temporal_sequence = np.expand_dims(np.array(temporal_frames), axis=0)
        temporal_prediction = temporal_model.predict(temporal_sequence)
        temporal_result = temporal_prediction[0][0]
    except Exception as e:
        print(f"Error during temporal analysis: {e}")
        return None

    # Frequency Analysis on the first frame
    try:
        gray_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        dft = cv2.dft(np.float32(gray_frame), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        freq_features = magnitude_spectrum.flatten()[:5000]
        freq_features = freq_features.reshape(1, -1)
        frequency_prediction = frequency_model.predict_proba(freq_features)
        frequency_result = frequency_prediction[0][1]
    except Exception as e:
        print(f"Error during frequency analysis: {e}")
        return None

    # Biometric Analysis
    try:
        blink_count = 0
        total_time = len(frames) / 30  # Assuming 30 FPS
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            if len(faces) > 0:
                landmarks = predictor(gray, faces[0])
                ear = compute_ear(landmarks)
                if ear < 0.21:
                    blink_count += 1
        blink_rate = blink_count / total_time if total_time > 0 else 0
        biometric_features = np.array([blink_rate]).reshape(1, -1)
        biometric_prediction = biometric_model.predict_proba(biometric_features)
        biometric_result = biometric_prediction[0][1]
    except Exception as e:
        print(f"Error during biometric analysis: {e}")
        return None

    # Combine results
    weights = {
        'spatial': 0.4,
        'temporal': 0.3,
        'frequency': 0.15,
        'biometric': 0.15
    }
    final_score = (
        weights['spatial'] * spatial_result +
        weights['temporal'] * temporal_result +
        weights['frequency'] * frequency_result +
        weights['biometric'] * biometric_result
    )

    threshold = 0.5  # Classification threshold

    if final_score >= threshold:
        classification = 'Deep Fake Video'
    else:
        classification = 'Real Video'

    # Output the final classification
    print(f"Video: {video_path}")
    print(f"Classification: {classification}")
    print(f"Confidence Score: {final_score:.2f}")
    return classification


# ===========================
# Flask API
# ===========================

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

@app.route('/')
def home():
    return "Hello, Deepfake project is running!"


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    video_file = request.files['file']
    upload_folder = 'upload'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)  # Create the upload folder if it doesn't exist

    video_path = os.path.join(upload_folder, video_file.filename)
    try:
        video_file.save(video_path)  # Save the uploaded video in the upload folder
        classification = analyze_video(video_path)  # Call the analysis function
        if classification is None:
            return jsonify({'error': 'Analysis failed'}), 500

        # Log the analysis result to the console
        print(f"Analysis Result for {video_file.filename}: {classification}")
        return jsonify({'classification': classification})
    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    if not os.path.exists('temp'):
        os.makedirs('temp')  # Create a temporary directory for uploaded files
    app.run(host='0.0.0.0', port=5000)