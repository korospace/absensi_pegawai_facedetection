# core
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO

# for croping face & create clf
from utils.helpers import preprocess_image
import face_recognition
import pickle
from PIL import Image
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import imgaug.augmenters as iaa
from mtcnn import MTCNN

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
    file        = request.files.get('photo')

    folder_path = os.path.join('assets', 'photo_frame', f"employee_{employee_id}")  # Path lengkap ke folder employee

    # Membuat folder jika belum ada
    if os.path.isfile("assets/model/trained_knn_model.clf"):
        os.remove("assets/model/trained_knn_model.clf")
        os.remove("assets/model/confusion_Matrix.png")
        
    os.makedirs(folder_path, exist_ok=True)

    # Mengubah file menjadi objek Image
    image = Image.open(file)
    image_name = os.path.splitext(file.filename)[0]

    num_images = 30

    # Buat direktori output jika belum ada
    output_dir = folder_path
    os.makedirs(output_dir, exist_ok=True)

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # Membalik gambar secara horizontal dengan peluang 0.5
        iaa.GaussianBlur(sigma=(0, 0.5)),  # Menambahkan efek blur dengan sigma antara 0 dan 0.5
        iaa.Affine(rotate=(-45, 45)),  # Memutar gambar antara -45 dan 45 derajat
        iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),  # Menambahkan noise Gaussian
        iaa.Multiply((0.8, 1.2), per_channel=0.2),  # Mengalikan intensitas warna dengan faktor antara 0.8 dan 1.2
        iaa.LinearContrast((0.75, 1.5)),  # Menambahkan kontras linear
        iaa.Crop(percent=(0, 0.1))  # Memotong sebagian gambar secara acak
    ], random_order=True)  # Mengatur urutan augmentasi secara acak

    num_images = int(num_images)

    detector = MTCNN()

    face_detected = False  # Flag untuk menandakan apakah wajah terdeteksi

    for i in range(num_images):
        augmented_image = seq.augment_image(np.array(image))
        augmented_image = augmented_image[:, :, :3]  # Menghilangkan saluran warna keempat (alpha channel)
        detections = detector.detect_faces(augmented_image)

        if len(detections) > 0:
            face_detected = True
            for j, detection in enumerate(detections):
                bounding_box = detection['box']
                face_image = augmented_image[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]]
                output_path = os.path.join(output_dir, f"{image_name}_{i}_{j}.jpg")
                Image.fromarray(face_image).save(output_path)
                print(f"Bagian wajah teraugmentasi disimpan: {output_path}")

    if not face_detected:
        shutil.rmtree(folder_path)  # Hapus folder parent
        return generate_response(message="No face detected", status=400)

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
