import os
import numpy as np
import streamlit as st
import pickle
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Define the path to the images
path = "/mount/src/presentation/AmazonProductImages"
os.chdir(path)

def extract_features(file, model):
    """ Function to load, preprocess, and normalize the image. """
    img = load_img(file, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape(1, 224, 224, 3)
    img = preprocess_input(img)
    features = model.predict(img)
    return features.squeeze()

# Initialize VGG16 model
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# Load or extract features for all images
pickle_file = "product_features.pkl"
if os.path.exists(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
else:
    data = {}
    products = [file for file in os.listdir(path) if file.endswith('.jpg')]
    for product in products:
        try:
            features = extract_features(product, model)
            data[product] = features
        except Exception as e:
            st.write(f"Error processing {product}: {e}")
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f)

# Ensure there are data points to process
if data:
    filenames = np.array(list(data.keys()))
    features = np.array(list(data.values()))

    if features.ndim == 1:  # Check if features need reshaping
        features = features.reshape(-1, 1)

    # Perform PCA
    pca = PCA(n_components=100, random_state=22)
    try:
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
            limited_files = filenames[:5]  # Display only up to 5 images per cluster
            images = []
            captions = []
            for file in limited_files:
                img = load_img(file, target_size=(224, 224))
                img_array = img_to_array(img)
                img_array = img_array / 255.0  # Normalize image data to [0, 1] range
                images.append(img_array)
                captions.append(file)

            cols = st.columns(len(images))  # Use Streamlit columns to display images
            for col, image, caption in zip(cols, images, captions):
                with col:
                    st.image(image, caption=caption, width=100)
    except ValueError as e:
        st.error(f"Error in PCA or KMeans processing: {e}")
else:
    st.error("No data available to process. Check your image directory or feature extraction.")
