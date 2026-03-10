import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

st.set_page_config(
    page_title="Prediksi Banjir Dayeuhkolot",
    layout="wide",
    initial_sidebar_state="expanded"
)

pemetaan_aliran = {
    # 1. Aliran Citarum (Utama) -> Data Acuan: Dayeuhkolot
    "Dayeuhkolot": "Dayeuhkolot",
    "Situ Cisanti (Kertasari)": "Dayeuhkolot",
    "Wangisagara (Majalaya)": "Dayeuhkolot",
    "Sapan (Titik Temu Anak Sungai)": "Dayeuhkolot",
    "Rancamanyar (Baleendah)": "Dayeuhkolot",
    "Nanjung (Margaasih)": "Dayeuhkolot",
    "Cabangbungin (Hilir Citarum)": "Dayeuhkolot",
    "Hantap": "Dayeuhkolot",

    # 2. Aliran Cisangkuy -> Data Acuan: Cipanas - Margamukti
    "Cipanas - Margamukti (Pangalengan)": "Cipanas - Margamukti",
    "Cileunca - Wanasari (Pangalengan)": "Cipanas - Margamukti",
    "Kertamanah - Margamukti (Pangalengan)": "Cipanas - Margamukti",
    "Kamasan (Banjaran)": "Cipanas - Margamukti",
    "Pataruman (Baleendah)": "Cipanas - Margamukti",
    "Arjasari": "Cipanas - Margamukti",

    # 3. Aliran Citarik & Cikeruh -> Data Acuan: Cikeruh - Jatiroke
    "Cikeruh - Jatiroke": "Cikeruh - Jatiroke",
    "Cicalengka (Termasuk Dampit)": "Cikeruh - Jatiroke",
    "Ciluluk - Cikancung": "Cikeruh - Jatiroke",
    "Rancaekek": "Cikeruh - Jatiroke",
    "Solokan Jeruk (Titik Citarik)": "Cikeruh - Jatiroke",
    "Mangalayang": "Cikeruh - Jatiroke",

    # 4. Aliran Ciwidey & Cisondari -> Data Acuan: Cisondari - Pasirjambu
    "Cisondari - Pasirjambu": "Cisondari - Pasirjambu",
    "Ciwidey": "Cisondari - Pasirjambu",
    "Cibeureum Sadu (Soreang)": "Cisondari - Pasirjambu",
    "Rancaupas": "Cisondari - Pasirjambu",

    # 5. Aliran Lainnya / Lokal -> Data Acuan: Bojongsoang
    "Bojongsoang": "Bojongsoang",
    "Cigede - Komplek Radio": "Bojongsoang",
    "Cijalupang - Peundeuy": "Bojongsoang",
    "Cipaku - Paseh": "Bojongsoang"
}

koordinat_stasiun = {
    "Dayeuhkolot": [-6.9881, 107.6281],
    "Cipanas - Margamukti": [-7.2185, 107.5565],
    "Cikeruh - Jatiroke": [-6.9450, 107.7680],
    "Cisondari - Pasirjambu": [-7.0680, 107.4780],
    "Bojongsoang": [-6.9740, 107.6400]
}

@st.cache_data
def load_data():
    file_mapping = {
        "Situ Cisanti": "Data Lengkap - cisanti (1).csv",
        "Kertasari": "Data Lengkap - kertasari (1).csv",
        "Cileunca": "Data Lengkap - cileunca (1).csv",
        "Kertamanah": "Data Lengkap - kertamanah (1).csv",
        "Cipanas": "Ciapanas - Margamukti.xlsx",
        "Hantap": "Hantap.xlsx",
        "Ciluluk": "CILULUK PASEH.csv",
        "Cisondari": "Cisondari.xlsx",
        "Cipaku Paseh": "Cipaku-Paseh.xlsx",
        "Bojongsoang": "Bojongsoang.csv",
        "Dayeuhkolot": "Data Banjir Daleuhlkolot - Sheet1.csv"
    }

    all_dfs = []

    for stasiun, nama_file in file_mapping.items():
        try:
            # Pengecekan Ekstensi File
            if nama_file.endswith('.xlsx'):
                df_temp = pd.read_excel(nama_file)
            else:
                # Blok Try-Except Berlapis "Anti-Error" untuk menangani Unicode pada CSV
                try:
                    df_temp = pd.read_csv(nama_file, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df_temp = pd.read_csv(nama_file, encoding='latin1')
                    except UnicodeDecodeError:
                        try:
                            df_temp = pd.read_csv(nama_file, encoding='utf-16')
                        except UnicodeDecodeError:
                            # Jurus terakhir: paksa baca dan abaikan karakter yang rusak
                            df_temp = pd.read_csv(nama_file, encoding='utf-8', encoding_errors='replace')

            df_temp.columns = df_temp.columns.str.strip()
            if "Banjir Ya/Tidak" not in df_temp.columns:
                for col in df_temp.columns:
                    if "banjir" in col.lower() and "tinggi" not in col.lower():
                        df_temp.rename(columns={col: "Banjir Ya/Tidak"}, inplace=True)
                        break
            df_temp["Kecamatan"] = stasiun

            all_dfs.append(df_temp)
        except FileNotFoundError:
            st.warning(f"⚠️ File '{nama_file}' untuk stasiun '{stasiun}' tidak ditemukan. Pastikan namanya benar dan ada di folder.")

    if not all_dfs:
        return pd.DataFrame(), {}

    df = pd.concat(all_dfs, ignore_index=True)

    df["Banjir Ya/Tidak"] = df["Banjir Ya/Tidak"].astype(str).str.strip().str.lower()
    mapping_target = {"ya": 1, "1": 1, "0": 0, "tidak": 0}
    df["Banjir Ya/Tidak"] = df["Banjir Ya/Tidak"].map(mapping_target)
    df = df.dropna(subset=["Banjir Ya/Tidak"])

    cols_numerik = ["Curah Hujan", "Debit Air", "Muka Air", "Tinggi Banjir"]
    for col in cols_numerik:
        if col in df.columns:
            # Ubah koma jadi titik (jika ada format indo), lalu ubah ke numerik
            df[col] = df[col].astype(str).str.replace("-", "0").str.replace(",", ".").str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Mapping Encoding Kecamatan ke Angka
    df["Kecamatan"] = df["Kecamatan"].astype(str).str.strip()
    stasiun_acuan_unik = sorted(list(set(pemetaan_aliran.values())))
    kec_mapping = {k: i for i, k in enumerate(stasiun_acuan_unik)}
    df["Kecamatan_Enc"] = df["Kecamatan"].map(kec_mapping).fillna(0)

    return df, kec_mapping

df, kecamatan_mapping = load_data()

if df.empty:
    st.error("File dataset tidak ditemukan! Pastikan file berada di folder yang sama.")
    st.stop()

@st.cache_resource
def train_model(dataframe):
    features = ["Kecamatan_Enc", "Curah Hujan", "Debit Air", "Muka Air", "Tinggi Banjir"]
    X = dataframe[features]
    y = dataframe["Banjir Ya/Tidak"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    jumlah_aman = (y_train == 0).sum()
    jumlah_banjir = (y_train == 1).sum()
    rasio_imbalance = jumlah_aman / jumlah_banjir if jumlah_banjir > 0 else 1

    model = XGBClassifier(
        n_estimators=100, 
        scale_pos_weight=rasio_imbalance,
        random_state=42, 
        learning_rate=0.1, 
        max_depth=4, 
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    akurasi = accuracy_score(y_test, y_pred)
    presisi = precision_score(y_test, y_pred, zero_division=0)
    recall_macro = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    recall_banjir = report['1']['recall'] if '1' in report else 0.0
    
    return model, features, akurasi, presisi, recall_macro, f1, cm, report, recall_banjir

# Memanggil fungsi training (saat pertama kali load)
model, features, akurasi, presisi, recall_macro, f1, cm, report, recall_banjir = train_model(df)

st.title("Sistem Peringatan Dini Banjir Berbasis Aliran Sungai")
st.markdown("Pantau dan prediksi potensi banjir di wilayah Kabupaten Bandung berdasarkan data hidrologis dan spasial.")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Prediksi Manual & Peta GIS", "Simulasi Real-time", "Performa Model AI"])

with st.sidebar:
    try:
        st.image("PHOTO-2026-03-10-09-11-05.jpg", use_container_width=True)
    except:
        pass
        
    st.header("Parameter Input")
    st.write("Masukkan data untuk dianalisis:")
    
    lokasi_select = st.selectbox("Pilih Lokasi (Kecamatan/Daerah)", options=list(pemetaan_aliran.keys()))
    curah_hujan = st.number_input("Curah Hujan (mm)", min_value=0.0, step=0.1)
    debit_air = st.number_input("Debit Air (m³/s)", min_value=0.0, step=0.1)
    muka_air = st.number_input("Tinggi Muka Air (m)", min_value=0.0, step=0.1)
    tinggi_banjir = st.number_input("Tinggi Genangan Air (m)", min_value=0.0, max_value=5.0, step=0.01)
    
    tombol_prediksi = st.button("🔍 Jalankan Prediksi", use_container_width=True, type="primary")

with tab1:
    col_hasil, col_peta = st.columns([1, 1.2]) 
    
    with col_hasil:
        st.subheader("Hasil Analisis")
        if tombol_prediksi:
            lokasi_utama = pemetaan_aliran[lokasi_select]
            kode_kec = kecamatan_mapping[lokasi_utama]
            
            input_data = pd.DataFrame([[kode_kec, curah_hujan, debit_air, muka_air, tinggi_banjir]], columns=features)
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            st.info(f"ℹ️ Titik analisis dialihkan ke data stasiun utama: **{lokasi_utama}**")
            
            if prediction == 1:
                st.error(f"⚠️ **POTENSI BANJIR TINGGI di {lokasi_select}**")
                st.progress(float(probability), text=f"Tingkat Bahaya / Probabilitas: {probability:.2%}")
            else:
                st.success(f"✅ **TIDAK ADA POTENSI BANJIR di {lokasi_select}**")
                st.progress(float(probability), text=f"Potensi Genangan / Probabilitas: {probability:.2%}")
                
            st.write("---")
            st.write("**Data yang diinput:**")
            st.write(f"- Curah Hujan: {curah_hujan} mm")
            st.write(f"- Debit Air: {debit_air} m³/s")
            st.write(f"- Tinggi Muka Air: {muka_air} m")
        else:
            st.info("👈 Silakan atur parameter di panel samping (Sidebar) dan tekan tombol 'Jalankan Prediksi'.")

    with col_peta:
        st.subheader("Peta Pantauan Sungai (GIS)")
        lokasi_utama_peta = pemetaan_aliran[lokasi_select]
        
        if HAS_FOLIUM:
            koor = koordinat_stasiun.get(lokasi_utama_peta, [-6.9881, 107.6281]) 
            
            m = folium.Map(location=koor, zoom_start=13, tiles="CartoDB positron")
            
            folium.Marker(
                koor, 
                popup=f"Stasiun Acuan: {lokasi_utama_peta}", 
                tooltip=f"Aliran Sungai {lokasi_utama_peta}",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)
            
            folium.Circle(
                location=koor,
                radius=1500, 
                color='crimson',
                fill=True,
                fill_color='crimson'
            ).add_to(m)

            st_folium(m, width=500, height=350, returned_objects=[])
        else:
            st.warning("Library 'folium' dan 'streamlit-folium' belum terinstal. Buka terminal dan jalankan `pip install folium streamlit-folium` untuk melihat peta.")

with tab2:
    st.subheader("Pantauan Sensor Virtual (Simulasi Real-time)")
    st.info(f"**Lokasi Pantauan Saat Ini:** {lokasi_select} \n\n*(Anda dapat mengubah lokasi pantauan melalui panel di sebelah kiri)*")
    
    if st.button("Cek Kondisi Terkini dari BMKG (Simulasi)"):
        skenario = np.random.choice(['Aman', 'Waspada', 'Bahaya'], p=[0.7, 0.2, 0.1])
        if skenario == 'Aman':
            sim_hujan, sim_debit, sim_muka, sim_tinggi = random.uniform(0, 10), random.uniform(20, 60), random.uniform(2.0, 4.5), 0.0
        elif skenario == 'Waspada':
            sim_hujan, sim_debit, sim_muka, sim_tinggi = random.uniform(10, 50), random.uniform(60, 100), random.uniform(4.5, 6.0), random.uniform(0.0, 0.3)
        else:
            sim_hujan, sim_debit, sim_muka, sim_tinggi = random.uniform(50, 110), random.uniform(100, 200), random.uniform(6.0, 8.0), random.uniform(0.3, 1.2)

        wib_now = datetime.utcnow() + timedelta(hours=7)
        jam_str = wib_now.strftime("%H:%M WIB")
        nama_lokasi = lokasi_select 
        lokasi_utama = pemetaan_aliran[nama_lokasi]
        kode_kec = kecamatan_mapping[lokasi_utama]
        
        input_sim = pd.DataFrame([[kode_kec, sim_hujan, sim_debit, sim_muka, sim_tinggi]], columns=features)
        pred_sim = model.predict(input_sim)[0]
        
        status_text, status_color, bg_color, icon = ("BERPOTENSI BANJIR", "#ff4b4b", "#ffebeb", "⚠️") if pred_sim == 1 else ("AMAN / TIDAK BANJIR", "#09ab3b", "#e8fdf0", "✅")
        
        st.markdown(f"""
        <div style="padding: 15px; border-radius: 10px; background-color: {bg_color}; border: 1px solid {status_color};">
            <h3 style="color: {status_color}; margin:0;">{icon} Status: {status_text}</h3>
            <p style="font-size: 16px; margin-top: 10px; color: #333;">
                <b>Kondisi Wilayah {nama_lokasi} (Aliran {lokasi_utama}) saat ini {status_text.lower()}.</b><br>
                Diperbarui pada: <b>{jam_str}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Curah Hujan", f"{sim_hujan:.1f} mm")
        k2.metric("Debit Air", f"{sim_debit:.1f} m³/s")
        k3.metric("Muka Air", f"{sim_muka:.2f} m")
        k4.metric("Tinggi Genangan", f"{sim_tinggi:.2f} m")

with tab3:
    st.subheader("Detail Evaluasi Algoritma XGBoost")
    target_color = "normal" if recall_banjir > 0.50 else "off"
    st.metric(label="🎯 Kemampuan Mendeteksi Banjir (Recall Kelas 1 - Target > 50%)", 
              value=f"{recall_banjir:.2%}", 
              delta="Target Tercapai!" if recall_banjir > 0.50 else "Masih di Bawah Target", 
              delta_color=target_color)
    st.markdown("---")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Akurasi Keseluruhan", f"{akurasi:.2%}")
    m2.metric("Precision", f"{presisi:.2%}")
    m3.metric("Recall (Rata-rata)", f"{recall_macro:.2%}")
    m4.metric("F1-Score", f"{f1:.2%}")
    
    st.divider()
    col_cm, col_rep = st.columns([1, 1.5])
    with col_cm:
        st.write("**Confusion Matrix:**")
        cm_df = pd.DataFrame(cm, index=['Aktual Tidak', 'Aktual Banjir'], columns=['Prediksi Tidak', 'Prediksi Banjir']) if cm.shape == (2, 2) else pd.DataFrame(cm)
        st.table(cm_df)
        
    with col_rep:
        st.write("**Detail Laporan Klasifikasi:**")
        st.dataframe(pd.DataFrame(report).transpose().style.format(precision=2))

    st.info("**Catatan Algoritma:** Model ini menggunakan **XGBoost** dengan parameter `scale_pos_weight` untuk mendeteksi data yang timpang (*imbalanced*).")
