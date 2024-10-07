import cv2
import os
import time

# Inisialisasi kamera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Lebar layar kamera
cam.set(4, 480)  # Tinggi layar kamera

# Path ke Haar Cascade
haarcascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(haarcascade_path)

# Input nama dan ID pengguna
name = input('\n Masukkan nama pengguna: ')
label_id = input('\n Masukkan ID pengguna (hanya angka): ')

# Pastikan label ID hanya angka
try:
    label_id = int(label_id)
except ValueError:
    print("[ERROR] ID harus berupa angka. Silakan jalankan ulang program dan masukkan ID yang benar.")
    cam.release()
    exit()

print("\n [INFO] Harap lihat ke kamera dan siapkan wajah Anda. Tekan tombol 's' untuk memulai pengambilan gambar...")

# Buat direktori dataset jika belum ada
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Inisialisasi variabel untuk menghitung gambar yang diambil
count = 0
capture_started = False

while True:
    ret, img = cam.read()
    if not ret:
        print("[ERROR] Gagal membuka kamera.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Tampilkan video dengan deteksi wajah
    cv2.imshow('image', img)

    # Tunggu pengguna menekan 's' untuk memulai pengambilan gambar
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and not capture_started:
        print("\n [INFO] Pengambilan gambar akan dimulai dalam 3 detik. Harap tetap stabil...")
        time.sleep(3)
        capture_started = True
        start_time = time.time()

    # Jika capture sudah dimulai, ambil 100 gambar
    if capture_started:
        # Ambil gambar setiap 0.1 detik (agar ada jeda antar gambar)
        if len(faces) > 0 and (time.time() - start_time) >= 0.1:
            for (x, y, w, h) in faces:
                count += 1
                filename = f"{name}_{label_id}_{count}.jpg"
                cv2.imwrite(os.path.join("dataset", filename), gray[y:y + h, x:x + w])
                print(f"[INFO] Gambar {count} disimpan.")

            start_time = time.time()  # Reset waktu untuk mengambil gambar berikutnya

        # Berhenti jika sudah mengambil 100 gambar
        if count >= 100:
            break

    # Tekan 'q' untuk keluar kapan saja
    if key == ord('q'):
        break

# Cleanup
print("\n [INFO] Mengambil gambar selesai.")
cam.release()
cv2.destroyAllWindows()
