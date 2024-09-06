from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
from model import get_pretrained_model
from utils import data_loader, image_preprocessing
from sklearn.model_selection import train_test_split
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the model
model = get_pretrained_model(num_classes=2)
model.load_state_dict(torch.load("cat_dog_classifier.pt"))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    if 'file' not in request.files:
        return render_template('predict.html', error='No file provided')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('predict.html', error='No selected file')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Open the image file
        image = Image.open(file_path)
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Run the model
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
        
        # Convert prediction to label
        label = 'dog' if predicted.item() == 1 else 'cat'
        
        return render_template('predict.html', prediction=label, filename=filename)

    return render_template('predict.html', error='Invalid file format')

@app.route('/accuracy', methods=['GET'])
def accuracy():
    # Load and preprocess data for accuracy evaluation
    image_list, label_list = data_loader("D:\\MLOps\\Cat-Dog-Classifier\\dataset")
    preprocessed_img = image_preprocessing(image_list)
    _, X_test, _, y_test = train_test_split(preprocessed_img, label_list, test_size=0.2, random_state=42)
    
    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test) * 100

    return render_template('accuracy.html', accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)