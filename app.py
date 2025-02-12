from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import torch
from flask import send_from_directory, abort
import faiss
import numpy as np
from PIL import Image
from torchvision import transforms
import os
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load pre-trained models
text_encoder = SentenceTransformer('clip-ViT-B-32')
image_encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights="ResNet50_Weights.IMAGENET1K_V1")
image_encoder.eval()  # Set model to evaluation mode

# Define image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
dataset_path = "test_data_v2"  # Ensure this path exists and contains images
if not os.path.exists(dataset_path):
    raise ValueError("The dataset path '{dataset_path}' does not exist. Please check the path.")

# Filter for valid image files
image_paths = [
    os.path.join(dataset_path, img) 
    for img in os.listdir(dataset_path) 
    if img.lower().endswith(('jpg', 'jpeg', 'png'))
]

if len(image_paths) == 0:
    raise ValueError("No valid image files found in '{dataset_path}'. Please ensure the folder contains .jpg, .jpeg, or .png files.")

# Function to process an image and extract its embedding
def process_image(image_path):
    try:
        # Open and preprocess the image
        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0)
        
        # Extract embedding using ResNet50
        with torch.no_grad():
            embedding = image_encoder(img_tensor).flatten().numpy()
        return embedding
    except Exception as e:
        print("Error processing {image_path}: {e}")
        return None

# Extract image embeddings
def extract_image_embeddings(image_paths):
    embeddings = []
    for path in image_paths:
        embedding = process_image(path)
        if embedding is not None:
            embeddings.append(embedding)
    return np.array(embeddings)

# Create FAISS index
image_embeddings = extract_image_embeddings(image_paths)

# Check if any valid embeddings were found
if image_embeddings.size == 0:
    raise ValueError("No valid image embeddings found. Please check the image paths and ensure they are valid.")

# Ensure image_embeddings is a 2D array for FAISS
if len(image_embeddings.shape) != 2:
    raise ValueError("Image embeddings must be a 2D array.")

# Initialize FAISS index
index = faiss.IndexFlatL2(image_embeddings.shape[1])
index.add(image_embeddings)
print("{index.ntotal} embeddings added to the FAISS index.")

# Route for image search
@app.route('/api/search', methods=['POST'])
def search_images():
    print("Request received")
    query = request.json.get('query', '')
    print("Query:", query)  # Log the query to verify it's received

    # Encode text query and ensure it's a 2D array for FAISS
    query_embedding = text_encoder.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)

    # Perform similarity search
    distances, indices = index.search(query_embedding, 5)  # Top-5 matches
    matched_images = [image_paths[i] for i in indices[0]]

    return jsonify({"results": matched_images})

# Serve images from the dataset folder
from flask import send_from_directory, abort
import os

@app.route('/test_data_v2/<path:filename>')
def serve_image(filename):
    directory = "test_data_v2"  # Ensure this is the correct directory
    file_path = os.path.join(directory, filename)

    # Check if the file exists and is a valid image
    if not os.path.isfile(file_path):
        abort(404)  # Return 404 if file is not found
        
        
        print("Matched images:", matched_images)


    return send_from_directory(directory, filename)


if __name__ == '__main__':
    app.run(debug=True)