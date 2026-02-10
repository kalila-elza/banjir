import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

@st.cache_data
def load_data():
    # Load data
    df = pd.read_csv("Data Banjir Daleuhlkolot - Sheet1.csv")
    
    # Bersihkan nama kolom dari spasi tambahan
    df.columns = df.columns.str.strip()

    # 1. Bersihkan kolom target (Banjir Ya/Tidak)
    df["Banjir Ya/Tidak"] = (
        df["Banjir Ya/Tidak"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    # Mapping target ke numerik
    mapping_target = {"ya": 1, "1": 1, "0": 0, "tidak": 0}
    df["Banjir Ya/Tidak"] = df["Banjir Ya/Tidak"].map(mapping_target)
    
    # Drop baris jika target tidak terdefinisi (NaN)
    df = df.dropna(subset=["Banjir Ya/Tidak"])

    # 2. Bersihkan fitur numerik (Ubah '-' menjadi 0)
    cols_numerik = ["Curah Hujan", "Debit Air", "Muka Air", "Tinggi Banjir"]
    for col in cols_numerik:
        if col in df.columns:
            # Ubah '-' menjadi '0', lalu konversi ke float
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("-", "0")
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 3. Encoding Kecamatan
    df["Kecamatan"] = df["Kecamatan"].astype(str).str.strip()
    kecamatan_list = df["Kecamatan"].unique().tolist()
    kec_mapping = {k: i for i, k in enumerate(kecamatan_list)}
    df["Kecamatan_Enc"] = df["Kecamatan"].map(kec_mapping)

    return df, kec_mapping

# Memanggil fungsi load data
df, kecamatan_mapping = load_data()

# Gunakan fitur yang sudah dibersihkan
features = ["Kecamatan_Enc", "Curah Hujan", "Debit Air", "Muka Air", "Tinggi Banjir"]
X = df[features]
y = df["Banjir Ya/Tidak"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y # Penting karena data banjir biasanya sedikit (imbalanced)
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.set_page_config(page_title="Prediksi Banjir Dayeuhkolot", layout="centered")
try:
    st.image("Dayeuhkolot.jpg", use_container_width=True)
except Exception as e:
    st.warning("⚠️ Gambar tidak ditemukan. Pastikan nama file 'daleuhkolot.jpy' benar atau ganti ekstensinya ke .jpg/.png")
    
st.title(" Prediksi Potensi Banjir")
st.write("Aplikasi ini memprediksi potensi banjir berdasarkan parameter lingkungan.")

# Tampilkan Statistik Model
col1, col2 = st.columns(2)
with col1:
    st.metric("Akurasi Model", f"{akurasi:.2%}")
with col2:
    st.write("**Confusion Matrix (TN, FP, FN, TP):**")
    st.write(cm)

st.divider()

st.subheader(" Input Data Lingkungan")

# Form Input
kecamatan_select = st.selectbox(
    "Pilih Kecamatan",
    options=list(kecamatan_mapping.keys())
)

c1, c2 = st.columns(2)
with c1:
    curah_hujan = st.number_input("Curah Hujan (mm)", min_value=0.0, step=0.1, help="Input intensitas hujan")
    debit_air = st.number_input("Debit Air (m³/s)", min_value=0.0, step=0.1)
with c2:
    muka_air = st.number_input("Tinggi Muka Air / TMA (m)", min_value=0.0, step=0.1)
    tinggi_banjir = st.number_input("Prediksi Tinggi Banjir (cm)", min_value=0.0, step=1.0)

# ==============================
# PROSES PREDIKSI
# ==============================
if st.button("🔍 Jalankan Prediksi", use_container_width=True):
    # Siapkan data input (urutan harus sama dengan fitur saat training)
    input_data = pd.DataFrame([[
        kecamatan_mapping[kecamatan_select],
        curah_hujan,
        debit_air,
        muka_air,
        tinggi_banjir
    ]], columns=features)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] # Peluang banjir

    st.subheader("Hasil Analisis:")
    if prediction == 1:
        st.error(f"⚠️ **POTENSI BANJIR TINGGI** (Probabilitas: {probability:.2%})")
        st.warning("Mohon waspada dan pantau informasi dari pihak terkait.")
    else:
        st.success(f"✅ **TIDAK ADA POTENSI BANJIR** (Probabilitas: {probability:.2%})")
        st.info("Kondisi saat ini diprediksi aman.")

# Sebelum cleaning
print(f"Total data awal: {len(df)}") 

# Setelah cleaning (drop NaN di target)
df = df.dropna(subset=["Banjir Ya/Tidak"])
print(f"Total data setelah cleaning: {len(df)}")
