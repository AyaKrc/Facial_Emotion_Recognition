from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
import cv2

# Initialize Flask app
app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model('final_model.h5')

# Define emotion labels
emotion_labels = ['fear', 'angry', 'sad', 'happy']

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route for images
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the uploaded image
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Preprocess the uploaded image
        image = Image.open(file_path).convert('RGB')  # Convert to RGB
        image = image.resize((224, 224))  # Resize to model input size
        image_array = img_to_array(image)  # Convert to array
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        image_array = image_array / 255.0  # Normalize to [0, 1]

        # Predict the emotion
        predictions = model.predict(image_array)
        emotion_index = np.argmax(predictions)  # Get the highest probability index
        emotion = emotion_labels[emotion_index]  # Map to emotion label

        return render_template('index.html', emotion=emotion, image_path=file_path)

    return redirect(request.url)

# Route for video-based emotion detection
@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        cap = cv2.VideoCapture(0)  # Access webcam

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess the frame
            resized_frame = cv2.resize(frame, (224, 224))
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            processed_frame = img_to_array(resized_frame)
            processed_frame = np.expand_dims(processed_frame, axis=0) / 255.0

            # Make prediction
            predictions = model.predict(processed_frame)
            emotion_index = np.argmax(predictions)
            emotion = emotion_labels[emotion_index]

            # Overlay emotion on the frame
            cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode the frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return app.response_class(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
