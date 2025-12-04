# Laporan Singkat: Implementasi Real-time rPPG

**Topik:** Deteksi Detak Jantung Non-Kontak (Webcam)
**Bahasa:** Python (OpenCV, MediaPipe, SciPy)

## Pembeda & Peningkatan Kualitas (Improvements)

### 1. ROI Spesifik pada Pipi (Skin Segmentation)

- **Implementasi Ini:** Menggunakan **MediaPipe Face Mesh** untuk menargetkan titik _landmark_ spesifik pada **Pipi Kiri dan Pipi Kanan**.

### 2. Visualisasi Sinyal Real-time (UI Informatif)

- **Implementasi Ini:** Menambahkan **Plot Grafik Sinyal** yang berjalan secara _real-time_ di layar[cite: 23].

### 3. Pemrosesan Real-time (Sliding Window)

- **Implementasi Ini:** Menggunakan mekanisme _buffer_ sirkular (_sliding window_)yang Memungkinkan estimasi detak jantung diperbarui secara terus-menerus pada setiap _frame_ tanpa jeda.
