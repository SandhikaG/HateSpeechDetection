from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, LSTMCell
import speech_recognition as sr
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import librosa
from pydub import AudioSegment  # For converting MP3 to WAV

app = Flask(__name__)

# Custom function to load model safely
def load_lstm_model(model_path):
    try:
        model = load_model(model_path, custom_objects={"LSTM": LSTM, "LSTMCell": LSTMCell}, compile=False)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# Load model and tokenizer
model = load_lstm_model('lstm_hatespeech_model.keras')

try:
    tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
    print("✅ Tokenizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")
    tokenizer = None

# Mapping numeric predictions to labels
label_mapping = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neutral'}

# Directory for saving audio files
audio_folder = 'audio_samples'
os.makedirs(audio_folder, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html', text="", prediction="", processed_text="")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tokenizer is None:
        return render_template('index.html', error="Model or tokenizer not loaded. Please check your files.")

    input_text = request.form.get('input_text', '').strip()
    audio_file = request.files.get('audio_file')

    if not input_text and not audio_file:
        return render_template('index.html', error="Please provide either text input or an audio file.")

    prediction_label = "No input provided"
    extracted_text = input_text  # Default to text input

    # Text Input Processing
    if input_text:
        try:
            max_len = 120  # Same as training
            text_seq = tokenizer.texts_to_sequences([input_text])
            text_padded = pad_sequences(text_seq, maxlen=max_len)
            text_prediction = np.argmax(model.predict(text_padded), axis=1)[0]
            prediction_label = label_mapping.get(text_prediction, "Unknown Prediction")
        except Exception as e:
            return render_template('index.html', error=f"Error processing text input: {e}")

    # Audio Processing
    if audio_file and audio_file.filename != '':
        try:
            filename = audio_file.filename
            file_ext = filename.split('.')[-1].lower()
            audio_path = os.path.join(audio_folder, filename)
            audio_file.save(audio_path)

            # Convert MP3 to WAV if needed
            if file_ext == 'mp3':
                wav_path = os.path.join(audio_folder, filename.replace('.mp3', '.wav'))
                AudioSegment.from_mp3(audio_path).export(wav_path, format='wav')
                os.remove(audio_path)  # Delete original MP3 file
                audio_path = wav_path  # Use converted WAV file

            # Convert Speech to Text
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                extracted_text = recognizer.recognize_google(audio_data)  # Convert to text

            os.remove(audio_path)  # Remove file after processing

            # Ensure text is available for prediction
            if extracted_text:
                max_len = 120  # Same as training
                text_seq = tokenizer.texts_to_sequences([extracted_text])
                text_padded = pad_sequences(text_seq, maxlen=max_len)
                prediction_audio = np.argmax(model.predict(text_padded), axis=1)[0]
                prediction_label = label_mapping.get(prediction_audio, "Unknown Prediction")

        except Exception as e:
            return render_template('index.html', error=f"Error processing audio file: {e}")

    return render_template('index.html', text=extracted_text, prediction=prediction_label, processed_text=extracted_text if audio_file else "")

if __name__ == '__main__':
    app.run(debug=True)
