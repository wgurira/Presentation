# Import necessary libraries
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 
from keras.applications.vgg16 import VGG16 
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import numpy as np
import streamlit as st
import pickle

# Define the path to the images
path = "/mount/src/presentation/AmazonProductImages"
os.chdir(path)

# Function to load, preprocess, and normalize the image
def extract_features(file, model):
    img = load_img(file, target_size=(224, 224))
    img = img_to_array(img)
    img = reshape(1, 224, 224, 3)
    img = preprocess_input(img)
    features = model.predict(img)
    return features.squeeze()

# Initialize VGG16 model
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# Load or extract features for all images
pickle_file = "product_features.pkl"
if os.path.exists(pickle_file):
    # Load features from the existing pickle file
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
else:
    # Extract features and save to pickle file
    data = {}
    products = [file for file in os.listdir(path) if file.endswith('.jpg')]
    for product in products:
        try:
            features = extract_features(product, model)
            data[product] = features
        except Exception as e:
            st.write(f"Error processing {product}: {e}")
    # Save the extracted features
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f)

# Load features from the data dictionary
filenames = np.array(list(data.keys()))
features = np.array(list(data.values()))

# Perform PCA
pca = PCA(n_components=100, random_state=22)
pca.fit(features)
features_pca = pca.transform(features)

# Cluster the PCA-transformed features
kmeans = KMeans(n_clusters=10, random_state=22)
kmeans.fit(features_pca)
groups = {i: [] for i in range(10)}
for file, cluster in zip(filenames, kmeans.labels_):
    groups[cluster].append(file)

# Streamlit UI to display clusters
st.title("Product Clustering")
for cluster_id, filenames in groups.items():
    st.header(f"Cluster {cluster_id+1}")
    # Display only up to 5 images per cluster
    limited_files = filenames[:5]  # Limit the number of files to 5
    images = []
    captions = []
    for file in limited_files:
        img = load_img(file, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize image data to [0, 1] range
        images.append(img_array)
        captions.append(file)

    # Use Streamlit columns to display images
    cols = st.columns(len(images))  # Creates a column for each image
    for col, image, caption in zip(cols, images, captions):
        with col:
            st.image(image, caption=caption, width=100)  # Adjust the width if needed
