from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
from yolov11.yolov11 import YOLOv11  # Import your YOLOv11 class
from gtts import gTTS

app = Flask(__name__)

# Directory for saving audio files
app.config['AUDIO_FOLDER'] = 'static/audio/'
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

detector = YOLOv11()

is_live = False

# Function to convert detected objects into speech
def text_to_speech(text, output_path):
    tts = gTTS(text)
    tts.save(output_path)

def generate_video():
    global is_live
    cap = cv2.VideoCapture(1) 

    last_detected = ""
    while is_live:
        ret, frame = cap.read()
        if not ret:
            break

        detected_objects = detector.detect_objects(frame)

        if detected_objects:
            detected_names = ', '.join(detected_objects)

            # Debug: Print detected objects to confirm detection
            print(f"Detected: {detected_names}")

            # If new objects are detected, convert them to speech
            if detected_names != last_detected:
                audio_file_path = os.path.join(app.config['AUDIO_FOLDER'], "output.mp3")
                text_to_speech(detected_names, audio_file_path)
                last_detected = detected_names

        # Convert the frame to JPEG format for MJPEG streaming
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()


# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to start live video feed
@app.route('/start_live', methods=['POST'])
def start_live():
    global is_live
    is_live = True  # Set live flag to True
    return redirect(url_for('live_view'))  # Redirect to live video view

# Route to view live video feed
@app.route('/live_view')
def live_view():
    return render_template('live_view.html')  # Separate template for live feed

# Route to handle live video feed stream
@app.route('/video_feed')
def video_feed():
    global is_live
    if is_live:
        return Response(generate_video(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return '', 204

# Route to stop live video feed
@app.route('/stop_live', methods=['POST'])
def stop_live():
    global is_live
    is_live = False  # Stop live feed
    return redirect(url_for('index'))

# Route to handle image uploads
@app.route('/upload_image', methods=['POST'])
def upload_image():
    file = request.files['image_file']
    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['AUDIO_FOLDER'], file.filename)
        file.save(file_path)

        # Process the image with YOLOv11
        img = cv2.imread(file_path)
        detected_objects = detector.detect_objects(img)
        
        if detected_objects:
            detected_names = ', '.join(detected_objects)
            
            # Convert detected objects to speech
            audio_file_path = os.path.join(app.config['AUDIO_FOLDER'], "output.mp3")
            text_to_speech(detected_names, audio_file_path)

            # Once detection is done, redirect to index.html with the audio ready
            return render_template('index.html', audio_file='audio/output.mp3')
        else:
            # If no objects detected, render the index with no detection
            return render_template('index.html', message="No objects detected.")
    return redirect(url_for('index'))


# Additional route for camera mode (optional, or same as live)
@app.route('/start_camera', methods=['POST'])
def start_camera():
    return redirect(url_for('live_view'))  # Redirect to live view (or handle differently)

if __name__ == '__main__':
    app.run(debug=True)
