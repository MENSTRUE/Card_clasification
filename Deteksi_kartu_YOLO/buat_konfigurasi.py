import os

# --- KONFIGURASI ---
# 1. Path ke folder training LAMA Anda (sumber nama kelas)
sumber_kelas_path = r'D:\00. codingan\AI\Uas Cnn Card\dataset\train'

# 2. Path ke folder proyek YOLO BARU Anda (tempat file akan dibuat)
proyek_yolo_path = r'D:\00. codingan\AI\Uas Cnn Card\Deteksi_kartu_YOLO'
# --- SELESAI KONFIGURASI ---


print("Memulai pembuatan file konfigurasi otomatis...")
print(f"Membaca nama kelas dari: {sumber_kelas_path}")

# --- Langkah 1: Dapatkan semua nama folder (kelas) dan urutkan ---
try:
    # Mengambil semua nama direktori dan mengurutkannya secara alfabetis
    class_names = sorted([d.name for d in os.scandir(sumber_kelas_path) if d.is_dir()])

    # Mengganti spasi dengan underscore (_) agar sesuai standar YOLO
    class_names_clean = [name.replace(' ', '_') for name in class_names]
except FileNotFoundError:
    print(f"\nERROR: Path sumber '{sumber_kelas_path}' tidak ditemukan!")
    print("Pastikan path ke dataset training lama Anda sudah benar.")
    exit()

if not class_names_clean:
    print("\nERROR: Tidak ada folder kelas yang ditemukan di path sumber.")
    exit()

print(f"Berhasil menemukan {len(class_names_clean)} kelas.")

# --- Langkah 2: Buat file classes.txt ---
classes_txt_path = os.path.join(proyek_yolo_path, 'classes.txt')
with open(classes_txt_path, 'w') as f:
    for name in class_names_clean:
        f.write(f"{name}\n")
print(f"✅ File 'classes.txt' berhasil dibuat di folder proyek YOLO.")

# --- Langkah 3: Buat file kartu.yaml ---
yaml_path = os.path.join(proyek_yolo_path, 'kartu.yaml')
# Menggunakan os.path.join agar path dataset sesuai untuk YAML dan sistem operasi
dataset_path_for_yaml = os.path.join(proyek_yolo_path, 'dataset').replace('\\', '/')

with open(yaml_path, 'w') as f:
    f.write(f"# File ini dibuat secara otomatis oleh skrip buat_konfigurasi.py\n")
    f.write(f"path: {dataset_path_for_yaml}\n")
    f.write(f"train: images\n")
    f.write(f"val: images\n")
    f.write(f"\n")
    f.write(f"# Daftar nama kelas ({len(class_names_clean)} kelas)\n")
    f.write(f"names:\n")
    for i, name in enumerate(class_names_clean):
        f.write(f"  {i}: {name}\n")  # Spasi di awal penting untuk format YAML
print(f"✅ File 'kartu.yaml' berhasil dibuat di folder proyek YOLO.")

print("\n--- SEMUA SELESAI ---")
print("File konfigurasi Anda sudah siap! Lanjutkan dengan mengisi folder dataset/images dan lakukan anotasi.")