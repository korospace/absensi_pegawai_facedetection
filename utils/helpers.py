import numpy as np
import cv2

def generate_response(message, status):

    if status == 200 or status == 201:
        status_bool = True
    else:
        status_bool = False

    res = {
        "status"  : status_bool,
        "message" : message,
    }

    return res, status

def preprocess_image(image):
     # Konversi gambar ke format BGR (OpenCV menggunakan format ini)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Perbaiki kecerahan dan kontras gambar
    alpha = 1.5  # Faktor kecerahan
    beta = 10  # Faktor kontras
    adjusted_image = cv2.convertScaleAbs(image_bgr, alpha=alpha, beta=beta)

    # Peningkatan kejelasan menggunakan kontras adaptif CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY))

    # Konversi gambar kembali ke format RGB setelah diproses
    enhanced_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)

    return enhanced_image_rgb

class KNNClassifier:
    def __init__(self, n_neighbors=9, weights='distance'):
        # Inisialisasi objek KNNClassifier.
        # Parameters:
        # - n_neighbors: Jumlah tetangga terdekat yang akan digunakan dalam klasifikasi (default: 9).
        # - weights: Metode pembobotan yang digunakan dalam klasifikasi (default: 'distance').
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y):
        # Melatih model KNN menggunakan data pelatihan.
        # Parameters:
        # - X: Data pelatihan dalam bentuk array numpy.
        # - y: Label data pelatihan dalam bentuk array numpy.
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def get_distances(self, X):
        # Menghitung jarak antara data uji (X) dengan data latih (self.X_train)
        distances = np.linalg.norm(self.X_train - X[:, np.newaxis], axis=2)

        # Membuat dictionary yang berisi parameter-parameter
        parameters = {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights
        }

        # Mengembalikan jarak dan parameter-parameter
        return distances, parameters

    def kneighbors(self, X, n_neighbors=None):
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        # Mencari tetangga terdekat dari data X.
        distances = []
        indices = []
        for x in X:
            dist = np.linalg.norm(self.X_train - x, axis=1)
            nearest_indices = np.argsort(dist)[:n_neighbors]
            distances.append(dist[nearest_indices])
            indices.append(nearest_indices)

        distances = np.array(distances)
        indices = np.array(indices)

        return distances, indices

    def predict(self, X):
        # Melakukan prediksi label berdasarkan data X.
        y_pred = []
        distances, indices = self.kneighbors(X)

        for dist, idx in zip(distances, indices):
            nearest_labels = self.y_train[idx]
            unique_labels, counts = np.unique(nearest_labels, return_counts=True)
            
            if self.weights == 'distance':
                # Jika metode pembobotan adalah 'distance', hitung pembobotan berdasarkan jarak
                weights = 1.0 / dist
                weighted_counts = np.tile(counts, (len(weights), 1)) * np.tile(weights, (len(counts), 1)).T
                y_pred.append(unique_labels[np.argmax(weighted_counts)])
            else:
                # Jika metode pembobotan bukan 'distance', pilih label yang paling sering muncul
                y_pred.append(unique_labels[np.argmax(counts)])
        
        return np.array(y_pred)