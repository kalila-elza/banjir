import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ==============================
# 1. KONFIGURASI HALAMAN
# ==============================
st.set_page_config(
    page_title="Prediksi Banjir Dayeuhkolot",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==============================
# 2. LOAD & CLEAN DATA
# ==============================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Data Banjir Daleuhlkolot - Sheet1.csv")
    except FileNotFoundError:
        st.error("File CSV tidak ditemukan! Pastikan file berada di folder yang sama.")
        return pd.DataFrame(), {}

    # Bersihkan nama kolom
    df.columns = df.columns.str.strip()

    # Bersihkan Target
    df["Banjir Ya/Tidak"] = df["Banjir Ya/Tidak"].astype(str).str.strip().str.lower()
    mapping_target = {"ya": 1, "1": 1, "0": 0, "tidak": 0}
    df["Banjir Ya/Tidak"] = df["Banjir Ya/Tidak"].map(mapping_target)
    df = df.dropna(subset=["Banjir Ya/Tidak"])

    # Bersihkan Fitur Numerik
    cols_numerik = ["Curah Hujan", "Debit Air", "Muka Air", "Tinggi Banjir"]
    for col in cols_numerik:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("-", "0").str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Encoding Kecamatan
    df["Kecamatan"] = df["Kecamatan"].astype(str).str.strip()
    kecamatan_list = df["Kecamatan"].unique().tolist()
    kec_mapping = {k: i for i, k in enumerate(kecamatan_list)}
    df["Kecamatan_Enc"] = df["Kecamatan"].map(kec_mapping)

    return df, kec_mapping

df, kecamatan_mapping = load_data()

if df.empty:
    st.stop()

# ==============================
# 3. TRAINING MODEL
# ==============================
features = ["Kecamatan_Enc", "Curah Hujan", "Debit Air", "Muka Air", "Tinggi Banjir"]
X = df[features]
y = df["Banjir Ya/Tidak"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi Singkat
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# ==============================
# 4. UI: HEADER & GAMBAR
# ==============================
try:
    st.image("Dayeuhkolot.jpg", use_container_width=True)
except:
    pass

st.title("Sistem Peringatan Dini Banjir")
st.markdown("---")

# ==============================
# 5. FITUR: KONDISI SAAT INI (REAL-TIME SIMULATION)
# ==============================
st.subheader("Kondisi Real-time (Simulasi)")

if st.button("Cek Kondisi Terkini dari BMKG (Simulasi)"):
    # 1. Generate Data Random
    skenario = np.random.choice(['Aman', 'Waspada', 'Bahaya'], p=[0.7, 0.2, 0.1])
    
    if skenario == 'Aman':
        sim_hujan = random.uniform(0, 10)
        sim_debit = random.uniform(20, 60)
        sim_muka  = random.uniform(2.0, 4.5)
        sim_tinggi = 0.0
    elif skenario == 'Waspada':
        sim_hujan = random.uniform(10, 50)
        sim_debit = random.uniform(60, 100)
        sim_muka  = random.uniform(4.5, 6.0)
        sim_tinggi = random.uniform(0.0, 0.3)
    else:
        sim_hujan = random.uniform(50, 110)
        sim_debit = random.uniform(100, 200)
        sim_muka  = random.uniform(6.0, 8.0)
        sim_tinggi = random.uniform(0.3, 1.2)
    
    # --- PERBAIKAN WAKTU DI SINI ---
    # Mengambil waktu UTC dan menambah 7 jam untuk WIB
    wib_now = datetime.utcnow() + timedelta(hours=7)
    
    # Format Bahasa Indonesia manual sederhana
    bulan_indo = {
        1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April', 5: 'Mei', 6: 'Juni',
        7: 'Juli', 8: 'Agustus', 9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
    }
    
    tgl_str = f"{wib_now.day} {bulan_indo[wib_now.month]} {wib_now.year}"
    jam_str = wib_now.strftime("%H:%M WIB")
    
    nama_kec = list(kecamatan_mapping.keys())[0]
    kode_kec = kecamatan_mapping[nama_kec]
    
    input_sim = pd.DataFrame([[
        kode_kec, sim_hujan, sim_debit, sim_muka, sim_tinggi
    ]], columns=features)
    
    pred_sim = model.predict(input_sim)[0]
    
    if pred_sim == 1:
        status_text = "BERPOTENSI BANJIR"
        status_color = "#ff4b4b" # Merah Streamlit
        bg_color = "#ffebeb"
        icon = "⚠️"
    else:
        status_text = "AMAN / TIDAK BANJIR"
        status_color = "#09ab3b" # Hijau Streamlit
        bg_color = "#e8fdf0"
        icon = "✅"
    
    st.markdown(f"""
    <div style="padding: 15px; border-radius: 10px; background-color: {bg_color}; border: 1px solid {status_color};">
        <h3 style="color: {status_color}; margin:0;">{icon} Status: {status_text}</h3>
        <p style="font-size: 16px; margin-top: 10px; color: #333;">
            <b>Kondisi Kecamatan {nama_kec} saat ini {status_text.lower()}.</b><br>
            Tanggal: <b>{tgl_str}</b> <br>
            Jam: <b>{jam_str}</b>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Curah Hujan", f"{sim_hujan:.1f} mm")
    k2.metric("Debit Air", f"{sim_debit:.1f} m³/s")
    k3.metric("Muka Air", f"{sim_muka:.2f} m")
    k4.metric("Tinggi Genangan", f"{sim_tinggi:.2f} m")

st.divider()

# ==============================
# 6. UI: INPUT MANUAL
# ==============================
st.subheader("Prediksi Manual")
st.write("Masukkan parameter di bawah ini untuk melakukan prediksi manual.")

with st.expander("Lihat Statistik Model"):
    col1, col2 = st.columns(2)
    col1.metric("Akurasi Model", f"{akurasi:.2%}")
    col2.write("Confusion Matrix:")
    col2.write(cm)

kecamatan_select = st.selectbox("Pilih Kecamatan", options=list(kecamatan_mapping.keys()))

c1, c2 = st.columns(2)
with c1:
    curah_hujan = st.number_input("Curah Hujan (mm)", min_value=0.0, step=0.1)
    debit_air = st.number_input("Debit Air (m³/s)", min_value=0.0, step=0.1)
with c2:
    muka_air = st.number_input("Tinggi Muka Air (m)", min_value=0.0, step=0.1)
    tinggi_banjir = st.number_input("Tinggi Genangan Air (m)", min_value=0.0, max_value=5.0, step=0.01)

if st.button("🔍 Jalankan Prediksi Manual", use_container_width=True):
    input_data = pd.DataFrame([[
        kecamatan_mapping[kecamatan_select],
        curah_hujan,
        debit_air,
        muka_air,
        tinggi_banjir
    ]], columns=features)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Hasil Analisis:")
    if prediction == 1:
        st.error(f"⚠️ **POTENSI BANJIR TINGGI** (Probabilitas: {probability:.2%})")
    else:
        st.success(f" **TIDAK ADA POTENSI BANJIR** (Probabilitas: {probability:.2%})")
