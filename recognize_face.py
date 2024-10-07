import cv2
import numpy as np
import os

# Path ke folder dataset
dataset_path = 'dataset'

# Membuat dictionary ID - Nama dari dataset
user_names = {}

# Iterasi melalui semua file dalam dataset untuk membuat mapping ID ke nama
image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
for image_file in image_files:
    try:
        # Ambil nama dan ID dari nama file
        name, id_str, _ = image_file.split("_")
        user_id = int(id_str)

        # Jika ID belum ada dalam dictionary, tambahkan
        if user_id not in user_names:
            user_names[user_id] = name
    except ValueError:
        print(f"[WARNING] Format nama file tidak sesuai, melewati file: {image_file}")
        continue

# Inisialisasi LBPH face recognizer dan load model yang sudah dilatih
recognizer = cv2.face.LBPHFaceRecognizer_create()
trainer_path = 'trainer/trainer.yml'

if os.path.exists(trainer_path):
    recognizer.read(trainer_path)
    print("[INFO] Model pengenalan wajah berhasil di-load.")
else:
    print("[ERROR] Model tidak ditemukan, harap latih model terlebih dahulu.")
    exit()

# Load Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inisialisasi kamera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Lebar layar kamera
cam.set(4, 480)  # Tinggi layar kamera

# Definisi font untuk label
font = cv2.FONT_HERSHEY_SIMPLEX

# Label untuk wajah yang tidak dikenali
unknown_label = "Tidak Dikenal"

print("\n[INFO] Menunggu pengenalan wajah... Tekan 'q' untuk keluar.")

while True:
    ret, img = cam.read()
    if not ret:
        print("[ERROR] Gagal membuka kamera.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Prediksi ID dari wajah yang terdeteksi
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # Jika confidence rendah, wajah dikenali
        if confidence < 50:
            user_name = user_names.get(id, unknown_label)
            confidence_text = f"{round(100 - confidence, 2)}%"  # Hitung tingkat kepercayaan
            label = f"{user_name} ({confidence_text})"
            color = (0, 255, 0)  # Warna hijau untuk wajah yang dikenali
        else:
            label = unknown_label
            color = (0, 0, 255)  # Warna merah untuk wajah yang tidak dikenali

        # Gambar kotak di sekitar wajah yang terdeteksi
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Tampilkan label di atas kotak
        cv2.putText(img, label, (x, y - 10), font, 0.8, color, 2)

    # Tampilkan hasil kamera dengan deteksi wajah
    cv2.imshow('Pengenalan Wajah', img)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
print("\n[INFO] Keluar dari program pengenalan wajah.")
cam.release()
cv2.destroyAllWindows()
