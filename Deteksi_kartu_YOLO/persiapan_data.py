import os
import shutil

# --- KONFIGURASI ---
# 1. Path ke folder dataset LAMA Anda (sumber gambar)
sumber_dataset_path = r'D:\00. codingan\AI\Uas Cnn Card\dataset'

# 2. Path ke folder dataset YOLO BARU Anda (tujuan gambar)
tujuan_images_path = r'D:\00. codingan\AI\Uas Cnn Card\Deteksi_kartu_YOLO\dataset\images'
# --- SELESAI KONFIGURASI ---

# Pastikan folder tujuan sudah ada
os.makedirs(tujuan_images_path, exist_ok=True)

print("Memulai proses pengumpulan gambar...")

# List folder yang akan diproses (train, valid, test)
folder_sumber = ['train', 'valid', 'test']
total_gambar_tercopy = 0

# Loop melalui setiap folder sumber (train, valid, test)
for folder in folder_sumber:
    path_folder_ini = os.path.join(sumber_dataset_path, folder)

    if not os.path.isdir(path_folder_ini):
        print(f"Peringatan: Folder '{path_folder_ini}' tidak ditemukan, akan dilewati.")
        continue

    print(f"\nMemproses folder: '{folder}'...")

    # Loop melalui setiap subfolder kelas di dalam (misal: 'ace_of_spades')
    for nama_kelas in os.listdir(path_folder_ini):
        path_kelas = os.path.join(path_folder_ini, nama_kelas)

        if os.path.isdir(path_kelas):
            # Loop melalui setiap file gambar di dalam folder kelas
            for nama_file in os.listdir(path_kelas):
                path_sumber_file = os.path.join(path_kelas, nama_file)

                # Cek apakah itu file dan merupakan gambar
                if os.path.isfile(path_sumber_file) and nama_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path_tujuan_file = os.path.join(tujuan_images_path, nama_file)

                    # Copy file ke folder tujuan
                    shutil.copy2(path_sumber_file, path_tujuan_file)
                    total_gambar_tercopy += 1

print(f"\n--- SEMUA SELESAI ---")
print(f"Total {total_gambar_tercopy} gambar berhasil disalin ke '{tujuan_images_path}'.")
print("Sekarang Anda siap untuk melakukan anotasi dengan LabelImg!")