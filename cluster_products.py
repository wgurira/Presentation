# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pickle

# Function to load and preprocess the image
def extract_features(file, model):
    img = load_img(file, target_size=(224, 224))
    img = np.array(img)
    reshaped_img = img.reshape(1, 224, 224, 3)
    imgx = preprocess_input(reshaped_img)
    features = model.predict(imgx)
    return features

# Load VGG16 model
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# Load images and extract features
path = "../AmazonProductImages"
os.chdir(path)
products = [file.name for file in os.scandir(path) if file.name.endswith('.jpg')]

data = {}
p = "product_features.pkl"

for product in products:
    try:
        feat = extract_features(product, model)
        data[product] = feat
    except Exception as e:
        st.write(f"Error processing {product}: {e}")

with open(p, 'wb') as file:
    pickle.dump(data, file)

# Load features and perform dimension reduction
data = pickle.load(open(p, 'rb'))
filenames = np.array(list(data.keys()))
feat = np.array(list(data.values()))
feat = feat.reshape(-1, 4096)

pca = PCA(n_components=100, random_state=22)
pca.fit(feat)
x = pca.transform(feat)

# Cluster feature vectors
kmeans = KMeans(n_clusters=10, random_state=22)
kmeans.fit(x)

# Organize clusters
groups = {}
for file, cluster in zip(filenames, kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
    groups[cluster].append(file)

# Streamlit UI to display clusters
st.title("Product Clustering")

selected_cluster = st.selectbox("Select a cluster to view:", list(range(10)))

if st.button("View Cluster"):
    plt.figure(figsize=(25, 25))
    files = groups[selected_cluster]
    for index, file in enumerate(files[:30]):
        plt.subplot(10, 5, index + 1)
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
    st.pyplot(plt)
