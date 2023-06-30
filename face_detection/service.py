# core
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO

# for croping face & create clf
import face_recognition
import pickle
from PIL import Image
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# for detect face
import cv2

# import utils
from utils.helpers import KNNClassifier
from utils.helpers import generate_response

# ============================
# Check Is Employe Have Folder
# ============================
def check_employee_folder(request):
    employee_id = request.args.get('employeeid')

    folder_path = os.path.join('assets', 'photo_frame', f"employee_{employee_id}")  # Path lengkap ke folder employee
    
    if os.path.exists(folder_path):
        return generate_response(message=f"Folder for employee {employee_id} exists.",status=200)
    else:
        return generate_response(message=f"Folder for employee {employee_id} does not exists.",status=404)

# ============================================
# Re Create .clf if new employee create folder
# ============================================
def create_model_file(request):
    
    employee_id = request.form.get('employeeid')
    files       = request.files.getlist('photo')

    folder_path = os.path.join('assets', 'photo_frame', f"employee_{employee_id}")  # Path lengkap ke folder employee

    # Membuat folder jika belum ada
    os.makedirs(folder_path, exist_ok=True)

    for file in files:
        filename = secure_filename(file.filename)
        image    = face_recognition.load_image_file(file)
        
        # Temukan lokasi wajah dalam gambar
        face_locations = face_recognition.face_locations(image)
        
        # if len(face_locations) > 0:
        if True:
            # Jika terdapat wajah dalam gambar
            for face_location in face_locations:
                # Crop bagian wajah dari gambar menggunakan koordinat wajah
                top, right, bottom, left = face_location
                face_image = image[top:bottom, left:right]
                
                # Simpan gambar wajah yang sudah di-crop
                face_filename  = f"crop_{filename}"
                face_save_path = os.path.join(folder_path, face_filename)
                pil_image = Image.fromarray(face_image)
                pil_image.save(face_save_path)
        else:
            # Jika tidak ada wajah dalam gambar
            shutil.rmtree(folder_path)  # Hapus folder parent
            return generate_response(message="no face", status=400)

    # Membuka folder assets/photo_frame
    encodings   = []
    employeesId = []

    employees_folder = os.listdir("assets/photo_frame")

    # Perulangan akan melalui setiap dataset dalam direktori pelatihan
    for employee_folder in employees_folder:
        employee_images = os.listdir(os.path.join("assets/photo_frame", employee_folder))

        # Loop melalui setiap gambar latih untuk orang yang saat ini
        for employee_image in employee_images:
            # Dapatkan enkode wajah untuk wajah di setiap file gambar
            face = face_recognition.load_image_file(os.path.join("assets/photo_frame", employee_folder, employee_image))

            # Anggap seluruh gambar adalah lokasi wajah
            height, width, _ = face.shape
            face_location    = (0, width, height, 0)
            face_enc = face_recognition.face_encodings(face, known_face_locations=[face_location])
            face_enc = np.array(face_enc)
            face_enc = face_enc.flatten()

            # Tambahkan enkode wajah untuk gambar saat ini dengan label yang sesuai (nama) ke data latihan
            encodings.append(face_enc)
            employeesId.append(employee_folder)

    # Bagi data menjadi set latihan dan pengujian
    uniqueId = np.unique(employeesId)
    
    encodings_train = []
    encodings_test  = []
    idTrain = []
    idTest  = []

    for id in uniqueId:
        name_encodings = [encoding for encoding, n in zip(encodings, employeesId) if n == id]
        name_labels    = [n for n in employeesId if n == id]
        encodings_train_value, encodings_test_value, idTrain_value, idTest_value = train_test_split(name_encodings, name_labels, test_size=0.3, random_state=42)
        encodings_train.extend(encodings_train_value)
        encodings_test.extend(encodings_test_value)
        idTrain.extend(idTrain_value)
        idTest.extend(idTest_value)

    # Buat dan latih klasifikasi KNN
    knn_clf = KNNClassifier(n_neighbors=9, weights='distance')
    knn_clf.fit(encodings_train, idTrain)

    # Evaluasi klasifikasi pada data pengujian
    predictions = knn_clf.predict(encodings_test)
    accuracy    = accuracy_score(idTest, predictions)
    precision   = precision_score(idTest, predictions, average='weighted')
    recall      = recall_score(idTest, predictions, average='weighted')
    report      = classification_report(idTest, predictions)

    # Cetak hasil Akurasi, Presisi, Recall, dan Laporan Klasifikasi
    print("Akurasi:", accuracy)
    print("Presisi:", precision)
    print("Recall:", recall)
    print("Laporan Klasifikasi:\n", report)
    print("Pelatihan selesai!")

    # Simpan klasifikasi KNN yang telah dilatih
    if "assets/model/trained_knn_model.clf" is not None:
        with open("assets/model/trained_knn_model.clf", 'wb') as f:
            pickle.dump(knn_clf, f)

    # Buat confusion matrix
    cm = confusion_matrix(idTest, predictions)
    class_names = np.unique(employeesId)

    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.savefig('assets/model/confusion_Matrix.png')

    return generate_response(message="Photos saved successfully.", status=200)

# ========================================
# Face detection for attendance validation
# ========================================
def compare_faces(request):
    employee_id = request.form.get('employeeid')
    photo       = request.files.get('photo')

    # Membaca konten file foto sebagai bytes
    photo_bytes = photo.read()

    # Membuat objek BytesIO dari bytes foto
    photo_bytesio = BytesIO(photo_bytes)

    accuracy = 0
    employee_folder_name = ""

    # load .clf
    with open("assets/model/trained_knn_model.clf", 'rb') as f:
        knn_clf = pickle.load(f)
        
        image = face_recognition.load_image_file(photo_bytesio)
        X_face_locations = face_recognition.face_locations(image)
        
        if len(X_face_locations) != 0:
            # Find encodings for faces in the test iamge
            faces_encodings = face_recognition.face_encodings(image, known_face_locations=X_face_locations)

            # Use the KNN model to find the best matches for the test face
            closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= 0.4 for i in range(len(X_face_locations))]
            predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
            lp = 0

            for name, (top, right, bottom, left) in predictions:
                accuracy = closest_distances[0][lp][0]
                employee_folder_name = name
                lp = lp + 1

    if(employee_folder_name == ""):
        return generate_response(message="no face", status=400)
    else:
        print(f"folder: {employee_folder_name}")
        if(f"employee_{employee_id}" == employee_folder_name):
            return generate_response(message=f"face confirmed. accuracy {accuracy}%", status=200)
        else:
            return generate_response(message="face mismatch", status=400)
