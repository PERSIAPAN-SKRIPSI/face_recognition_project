import cv2
import numpy as np
import os
from PIL import Image

# Path ke folder dataset
path = 'dataset'

# Inisialisasi LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load Haar Cascade untuk deteksi wajah
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Fungsi untuk mendapatkan gambar dan ID dari dataset
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    face_samples = []
    ids = []

    for image_path in image_paths:
        try:
            # Ambil ID dari nama file gambar
            filename = os.path.split(image_path)[-1]
            name, id_str, _ = filename.split("_")
            id = int(id_str)  # Konversi ke integer

            # Convert gambar ke grayscale menggunakan Pillow
            PIL_img = Image.open(image_path).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')

            # Deteksi wajah di gambar
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)

        except ValueError as ve:
            print(f"[WARNING] Format nama file tidak sesuai, melewati file: {image_path}")
            continue
        except Exception as e:
            print(f"[ERROR] Terjadi kesalahan pada file: {image_path}, Kesalahan: {e}")
            continue

    return face_samples, ids


# Dapatkan wajah dan ID dari dataset
print("\n [INFO] Melatih model wajah. Tunggu sebentar...")
faces, ids = get_images_and_labels(path)
recognizer.train(faces, np.array(ids))

# Simpan model ke dalam file
if not os.path.exists('trainer'):
    os.makedirs('trainer')
recognizer.write('trainer/trainer.yml')  # Model tersimpan di folder trainer
print(f"\n [INFO] {len(np.unique(ids))} wajah berhasil dilatih.")
