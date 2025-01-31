from flask import Flask, request, render_template, jsonify
import cv2
import os
from werkzeug.utils import secure_filename
import numpy as np

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def make_sketch(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inverted = 255 - gray

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)

    # Dodge blend effect
    def dodgeV2(image, mask):
        return cv2.divide(image, 255 - mask, scale=256)

    sketch = dodgeV2(gray, blurred)

    return sketch

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/sketch', methods=['POST'])
def sketch():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Read image
        img = cv2.imread(file_path)
        sketch_img = make_sketch(img)

        # Save sketch image
        sketch_img_name = filename.rsplit('.', 1)[0] + "_sketch.jpg"
        sketch_img_path = os.path.join(app.config['UPLOAD_FOLDER'], sketch_img_name)
        cv2.imwrite(sketch_img_path, sketch_img)

        return render_template('home.html', org_img_name=filename, sketch_img_name=sketch_img_name)
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
