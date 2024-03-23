import os
import numpy as np
import librosa
from keras.models import load_model

# Function to extract features from audio files
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate).T, axis=0)
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    return np.concatenate((mfccs, chroma, mel))

# Load the pre-trained model
model = load_model('baby_cry_model.h5')

# Function to predict emotion from audio file
def predict_emotion(audio_file):
    # Extract features from audio file
    features = extract_features(audio_file)
    # Reshape features to match model input shape
    features = np.expand_dims(features, axis=0)
    # Predict emotion label
    prediction = model.predict(features)
    # Get the predicted label index
    predicted_index = np.argmax(prediction)
    # Map index to emotion label
    emotion_labels = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
    predicted_emotion = emotion_labels[predicted_index]
    return predicted_emotion

# Test the prediction function with an audio file
audio_file_path = './donateacry_corpus_cleaned_and_updated_data/discomfort/10A40438-09AA-4A21-83B4-8119F03F7A11-1430925142-1.0-f-26-dc.wav'  # Change this to your audio file path
predicted_emotion = predict_emotion(audio_file_path)
print(f"Predicted Emotion: {predicted_emotion}")
