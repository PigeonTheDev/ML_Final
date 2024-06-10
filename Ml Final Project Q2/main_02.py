# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

file_path = 'Dataset ML.csv' 
df = pd.read_csv(file_path, encoding='latin1')

categorical_cols = list(df.columns[:155])
numeric_cols = list(df.columns[155:])


# Preprocessing pipeline for numeric columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

# Preprocessing pipeline for categorical columns
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Preprocess the data
df_preprocessed = preprocessor.fit_transform(df)


# Apply PCA for dimensionality reduction
pca = PCA(n_components=10)  # Adjust the number of components based on explained variance
df_pca = pca.fit_transform(df_preprocessed)


# Determine a suitable eps value for DBSCAN using the k-nearest neighbors method
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(df_pca)
distances, indices = neighbors_fit.kneighbors(df_pca)

# Sort the distances and plot them
distances = np.sort(distances[:, 4], axis=0)
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.xlabel('Data Points sorted by distance')
plt.ylabel('4th Nearest Neighbor Distance')
plt.title('Elbow Method to determine eps for DBSCAN')
plt.show()

# Apply DBSCAN with the identified eps value
eps_value = 3.5  # Replace with the chosen value from the plot
dbscan = DBSCAN(eps=eps_value, min_samples=5)
dbscan_labels = dbscan.fit_predict(df_pca)


dbscan_silhouette = silhouette_score(df_pca, dbscan_labels)
print('\033[92m' + f'DBSCAN Silhouette Coefficient with eps={eps_value}: {dbscan_silhouette}' + '\033[0m')


# K-means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(df_pca)
kmeans_silhouette = silhouette_score(df_pca, kmeans_labels)
print('\033[92m' f'K-means Silhouette Coefficient: {kmeans_silhouette}' + '\033[0m')

# Deep Neural Clustering (Autoencoder-based)
# Define the autoencoder model
input_dim = df_pca.shape[1]
encoding_dim = 128

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the autoencoder
autoencoder.fit(df_pca, df_pca, epochs=50, batch_size=256, shuffle=True, validation_split=0.2, verbose=0)

# Get the encoded data
encoded_data = encoder.predict(df_pca)

# Perform K-means clustering on the encoded data
kmeans_encoded = KMeans(n_clusters=5, random_state=42)
kmeans_encoded_labels = kmeans_encoded.fit_predict(encoded_data)

# Calculate the Silhouette Coefficient
deep_clustering_silhouette = silhouette_score(df_pca, kmeans_encoded_labels)
print('\033[92m' + f'Deep Clustering Silhouette Coefficient: {deep_clustering_silhouette}'+ '\033[0m')
