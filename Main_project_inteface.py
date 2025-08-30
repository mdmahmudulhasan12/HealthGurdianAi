from flask import Flask, render_template_string, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# ----------------- Flask App -----------------
app = Flask(__name__)

# ----------------- Load Models -----------------
brain_model = load_model('models/model.h5')        # Brain tumor model
skin_model = load_model('skin_cancer_cnn.h5')      # Skin cancer model (root folder)

# ----------------- Labels -----------------
brain_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# ----------------- Uploads Folder -----------------
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ----------------- Helper Functions -----------------
def predict_brain_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = brain_model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if brain_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {brain_labels[predicted_class_index]}", confidence_score


def predict_skin_cancer(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = skin_model.predict(img_array)
    class_label = "Malignant" if prediction > 0.5 else "Benign"

    return class_label, float(prediction[0][0])

# ----------------- Load HTML -----------------
with open("index.html", "r", encoding="utf-8") as f:
    index_html = f.read()

# ----------------- Routes -----------------
@app.route('/', methods=['GET', 'POST'])
def index():
    result, confidence, file_path = None, None, None

    if request.method == 'POST':
        model_type = request.form.get('model_type')  # "brain" or "skin"
        file = request.files['file']

        if file:
            # Save uploaded file
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            # Predict based on selected model
            if model_type == "brain":
                result, confidence = predict_brain_tumor(file_location)
            elif model_type == "skin":
                result, confidence = predict_skin_cancer(file_location)

            file_path = f'/uploads/{file.filename}'

    return render_template_string(index_html,
                                  result=result,
                                  confidence=f"{confidence*100:.2f}%" if confidence else None,
                                  file_path=file_path)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ----------------- Run App -----------------
if __name__ == '__main__':
    app.run(debug=True, port=8080)
