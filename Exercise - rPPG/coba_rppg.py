import cv2
import numpy as np
import mediapipe as mp
import time
from scipy import signal

class RealTimeRPPG:
    def __init__(self, video_source=0, buffer_size=150, fps=30):
        # Inisialisasi Webcam
        self.cap = cv2.VideoCapture(video_source)
        
        # Konfigurasi Parameter Sinyal
        self.buffer_size = buffer_size
        self.fps = fps
        self.signal_buffer = []
        self.bpm = 0
        
        # Inisialisasi MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # --- PERUBAHAN DISINI: DEFINISI ROI PIPI SAJA ---
        # Kami memisahkan pipi kiri dan kanan menggunakan indeks landmark MediaPipe yang spesifik.
        # Area ini berada di bawah mata dan di samping hidung (area paling vaskular).
        
        self.left_cheek_landmarks = [116, 117, 118, 100, 126, 209, 49, 50]
        self.right_cheek_landmarks = [345, 346, 347, 329, 355, 429, 279, 280]
    

    def get_roi_average(self, frame, landmarks):
        """
        Ekstraksi rata-rata kanal Hijau (Green) hanya dari area PIPI KIRI dan PIPI KANAN.
        """
        h, w, _ = frame.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Helper function untuk mendapatkan titik dari indeks
        def get_points(indices):
            pts = []
            for idx in indices:
                pt = landmarks.landmark[idx]
                pts.append((int(pt.x * w), int(pt.y * h)))
            return np.array(pts, np.int32)

        # 1. Ambil koordinat Pipi Kiri & Pipi Kanan
        pts_left = get_points(self.left_cheek_landmarks)
        pts_right = get_points(self.right_cheek_landmarks)
        
        # 2. Gambar area tersebut ke dalam Mask (warna putih)
        cv2.fillConvexPoly(mask, pts_left, 255)
        cv2.fillConvexPoly(mask, pts_right, 255)
        
        # 3. Hitung rata-rata kanal Hijau HANYA pada area yang dimasker (Pipi)
        # cv2.mean mengembalikan (B, G, R, Alpha), kita ambil index 1 untuk Green
        mean_val = cv2.mean(frame, mask=mask)[1] 
        
        # Kembalikan nilai rata-rata dan titik-titik untuk visualisasi
        return mean_val, [pts_left, pts_right]

    def process_signal(self, raw_signal):
        """
        Sama seperti sebelumnya: Bandpass Filter (0.67 - 4.0 Hz) & Detrending
        """
        if len(raw_signal) < self.buffer_size:
            return raw_signal
            
        detrended = signal.detrend(raw_signal)
        
        nyquist = 0.5 * self.fps
        low = 0.67 / nyquist
        high = 4.0 / nyquist
        b, a = signal.butter(3, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, detrended)
        
        return filtered

    def calculate_bpm(self, filtered_signal):
        """
        Estimasi BPM menggunakan FFT
        """
        if len(filtered_signal) < self.buffer_size:
            return 0
            
        window = np.hamming(len(filtered_signal))
        signal_windowed = filtered_signal * window
        
        fft_res = np.fft.rfft(signal_windowed)
        freqs = np.fft.rfftfreq(len(signal_windowed), 1.0/self.fps)
        
        valid_idx = np.where((freqs >= 0.67) & (freqs <= 4.0))[0]
        if len(valid_idx) == 0:
            return 0
            
        valid_fft = np.abs(fft_res[valid_idx])
        peak_idx = np.argmax(valid_fft)
        dominant_freq = freqs[valid_idx[peak_idx]]
        
        bpm = dominant_freq * 60.0
        return bpm

    def draw_signal_plot(self, frame, signal_data):
        h, w, _ = frame.shape
        plot_h, plot_w = 100, 300
        x_offset, y_offset = w - plot_w - 20, 50
        
        cv2.rectangle(frame, (x_offset, y_offset), (x_offset+plot_w, y_offset+plot_h), (0,0,0), -1)
        
        if len(signal_data) > 2:
            norm_sig = np.array(signal_data[-plot_w:])
            if np.max(norm_sig) != np.min(norm_sig):
                norm_sig = (norm_sig - np.min(norm_sig)) / (np.max(norm_sig) - np.min(norm_sig))
            else:
                norm_sig = np.zeros_like(norm_sig)
                
            points = []
            for i, val in enumerate(norm_sig):
                x = x_offset + i
                y = y_offset + plot_h - int(val * plot_h)
                points.append((x, y))
                
            cv2.polylines(frame, [np.array(points, np.int32)], False, (0, 255, 0), 2)
            
        cv2.putText(frame, "Cheek ROI Signal", (x_offset, y_offset - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def run(self):
        print("Mulai mengambil data rPPG dari PIPI... Tekan 'q' untuk berhenti.")
        prev_time = time.time()
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            current_time = time.time()
            fps_real = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Ambil rata-rata dari kedua pipi
                    g_mean, roi_polys = self.get_roi_average(frame, face_landmarks)
                    
                    self.signal_buffer.append(g_mean)
                    if len(self.signal_buffer) > self.buffer_size:
                        self.signal_buffer.pop(0)
                        
                    processed_signal = self.process_signal(np.array(self.signal_buffer))
                    
                    if len(self.signal_buffer) == self.buffer_size:
                        self.bpm = self.calculate_bpm(processed_signal)
                    
                    # Visualisasi: Gambar kotak di kedua pipi
                    for poly in roi_polys:
                        cv2.polylines(frame, [poly], True, (255, 0, 0), 2)
                    
                    self.draw_signal_plot(frame, processed_signal)

            cv2.putText(frame, f"BPM: {self.bpm:.1f}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"FPS: {fps_real:.1f}", (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            cv2.imshow('Real-time rPPG (Cheeks Only)', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = RealTimeRPPG()
    app.run()