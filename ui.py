import streamlit as st
import cv2
from backend import PersonalityPredictor
import matplotlib.pyplot as plt
import pandas as pd

# Konfigurasi halaman agar lebih lebar
st.set_page_config(page_title="Sistem Cerdas Kepribadian", layout="wide")

st.title("🧠 Sistem Identifikasi Kepribadian via Ekspresi Wajah")
st.markdown("---")

# Inisialisasi predictor dalam session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = PersonalityPredictor()
if 'run' not in st.session_state:
    st.session_state.run = False

# Membuat dua kolom utama
col_video, col_analysis = st.columns([1.2, 1])

with col_video:
    st.subheader("📷 Kamera Real-Time")
    image_place = st.empty() # Placeholder untuk video
    
    # Tombol kontrol di bawah kamera
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Mulai/Stop Kamera", use_container_width=True):
            st.session_state.run = not st.session_state.run
    with btn_col2:
        if st.button("Reset Data", use_container_width=True):
            st.session_state.predictor.reset()
            st.rerun()

with col_analysis:
    st.subheader("📊 Hasil Analisis Big Five")
    chart_place = st.empty() # Placeholder untuk grafik
    table_place = st.empty() # Placeholder untuk tabel

# Logika penayangan video dan update analisis otomatis
if st.session_state.run:
    cap = cv2.VideoCapture(0)
    
    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal mengakses kamera.")
            break
        
        # 1. Proses deteksi emosi
        emotion, region = st.session_state.predictor.analyze_frame(frame)
        
        # 2. Gambar kotak ROI (MediaPipe logic via Backend)
        if region:
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Emosi: {emotion}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 3. Update Video di Frame Kiri
        image_place.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 4. Update Analisis secara Real-Time di Frame Kanan
        scores = st.session_state.predictor.calculate_personality_scores()
        if scores:
            # Update Grafik
            fig, ax = plt.subplots(figsize=(5, 4))
            colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#C2C2F0']
            ax.bar(scores.keys(), scores.values(), color=colors)
            ax.set_ylim(0, 100)
            plt.xticks(rotation=45)
            plt.tight_layout()
            chart_place.pyplot(fig)
            plt.close(fig) # Membersihkan memori
            
            # Update Tabel
            df_scores = pd.DataFrame(list(scores.items()), columns=['Dimensi', 'Skor (%)'])
            table_place.table(df_scores.set_index('Dimensi'))

    cap.release()
else:
    # Tampilan saat kamera mati
    image_place.info("Kamera dalam posisi OFF. Klik 'Mulai Kamera' untuk memulai analisis.")