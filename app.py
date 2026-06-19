import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.metrics import geometric_mean_score
import matplotlib.pyplot as plt

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

# --- FUNGSI LOGIKA DARI XGBoost.ipynb ---
def tentukan_level_banjir(tma):
    """
    Menentukan level banjir berdasarkan Tinggi Muka Air (TMA) sesuai notebook.
    Threshold:
    - Normal  : < 0.57 m
    - Waspada : 0.57 - 0.93 m
    - Siaga   : 0.93 - 1.30 m
    - Awas    : > 1.30 m
    """
    if tma < 0.57:
        return '0 - Normal'
    elif 0.57 <= tma < 0.93:
        return '1 - Waspada (Siaga 3)'
    elif 0.93 <= tma <= 1.30:
        return '2 - Siaga (Siaga 2)'
    else:
        return '3 - Awas (Siaga 1)'


def _normalisasi_nama(s):
    """Normalisasi nama stasiun (huruf kecil, hapus spasi & tanda hubung) supaya variasi
    penulisan seperti 'Cisondari - Pasirjambu' vs 'Cisondari-Pasir Jambu' tetap dikenali sebagai
    stasiun yang sama."""
    return str(s).lower().replace(' ', '').replace('-', '')


def cocokkan_kecamatan(daftar_kelas, nama_target):
    """Mencocokkan nama 'aliran induk' (dari pemetaan_aliran) ke salah satu nama Kecamatan
    yang benar-benar ada di dataset CSV. Dilakukan bertahap: cocok persis (setelah normalisasi),
    lalu cocok sebagian (substring), lalu fallback ke 'Dayeuhkolot' (data paling lengkap).
    Mengembalikan nama_kecamatan_terpilih."""
    daftar_kelas = list(daftar_kelas)
    target_norm = _normalisasi_nama(nama_target)

    for c in daftar_kelas:
        if _normalisasi_nama(c) == target_norm:
            return c
    for c in daftar_kelas:
        c_norm = _normalisasi_nama(c)
        if target_norm in c_norm or c_norm in target_norm:
            return c
    if "Dayeuhkolot" in daftar_kelas:
        return "Dayeuhkolot"
    return daftar_kelas[0] if daftar_kelas else None


def get_level_style(label):
    """Mapping ikon, warna, dan deskripsi singkat untuk tiap level banjir,
    dipakai untuk tampilan ala widget cuaca pada tab Prakiraan 7 Hari."""
    if "Awas" in label:
        return {"icon": "⛈️", "color": "#ff5c5c", "bg": "#3a1414", "short": "Awas", "desc": "Evakuasi segera"}
    elif "Siaga" in label:
        return {"icon": "🌧️", "color": "#ff9f43", "bg": "#3a2814", "short": "Siaga", "desc": "Waspada tinggi"}
    elif "Waspada" in label:
        return {"icon": "⛅", "color": "#f5d547", "bg": "#3a3414", "short": "Waspada", "desc": "Pantau terus"}
    else:
        return {"icon": "☀️", "color": "#4dd07a", "bg": "#143a1f", "short": "Normal", "desc": "Kondisi aman"}


pemetaan_aliran = {
    # 1. Aliran Citarum (Utama)
    "Dayeuhkolot": "Dayeuhkolot",
    "Situ Cisanti (Hulu Citarum, Kertasari)": "Dayeuhkolot",
    "Cisanti": "Dayeuhkolot", 
    "Kertasari": "Dayeuhkolot",
    "Wangisagara (Majalaya)": "Dayeuhkolot",
    "Majalaya": "Dayeuhkolot",
    "Sapan (Titik temu beberapa anak sungai)": "Dayeuhkolot",
    "Rancamanyar (Baleendah)": "Dayeuhkolot",
    "Nanjung (Margaasih)": "Dayeuhkolot",
    "Cabangbungin (Hilir Citarum)": "Dayeuhkolot",
    "Hantap": "Dayeuhkolot",

    # 2. Aliran Cisangkuy
    "Cipanas - Margamukti (Pangalengan)": "Cipanas - Margamukti",
    "Cipanas": "Cipanas - Margamukti",
    "Cileunca - Wanasari (Pangalengan)": "Cipanas - Margamukti",
    "Cileunca": "Cipanas - Margamukti",
    "Kertamanah - Margamukti (Pangalengan)": "Cipanas - Margamukti",
    "Kertamanah": "Cipanas - Margamukti",
    "Kamasan (Banjaran)": "Cipanas - Margamukti",
    "Pataruman (Baleendah)": "Cipanas - Margamukti",
    "Arjasari": "Cipanas - Margamukti",

    # 3. Aliran Citarik & Cikeruh
    "Cikeruh - Jatiroke": "Cikeruh - Jatiroke",
    "Jatiroke": "Cikeruh - Jatiroke",
    "Cicalengka (Termasuk titik Dampit)": "Cikeruh - Jatiroke",
    "Ciluluk - Cikancung": "Cikeruh - Jatiroke",
    "Ciluluk": "Cikeruh - Jatiroke",
    "Rancaekek": "Cikeruh - Jatiroke",
    "Solokan Jeruk (Titik Citarik)": "Cikeruh - Jatiroke",
    "Mangalayang": "Cikeruh - Jatiroke",

    # 4. Aliran Ciwidey & Cisondari
    "Cisondari - Pasirjambu": "Cisondari - Pasirjambu",
    "Cisondari": "Cisondari - Pasirjambu",
    "Ciwidey": "Cisondari - Pasirjambu",
    "Cibeureum Sadu (Soreang)": "Cisondari - Pasirjambu",
    "Rancaupas": "Cisondari - Pasirjambu",

    # 5. Aliran Lainnya / Lokal
    "Bojongsoang": "Bojongsoang",
    "Cigede - Komplek Radio (Bojongsoang)": "Bojongsoang",
    "Cijalupang - Peundeuy": "Bojongsoang",
    "Cipaku - Paseh": "Bojongsoang",
    "Cipaku Paseh": "Bojongsoang" 
}

koordinat_stasiun = {
    "Dayeuhkolot": [-6.9881, 107.6281],
    "Cipanas - Margamukti": [-7.2185, 107.5565],
    "Cikeruh - Jatiroke": [-6.9450, 107.7680],
    "Cisondari - Pasirjambu": [-7.0680, 107.4780],
    "Bojongsoang": [-6.9740, 107.6400]
}

# --- SIDEBAR ---
with st.sidebar:
    try:
        st.image("Dayeuhkolot.jpg", use_container_width=True)
    except:
        pass
        
    st.header("🎛️ Parameter Input")
    st.write("Masukkan data untuk dianalisis:")
    
    lokasi_select = st.selectbox("Pilih Lokasi (Kecamatan/Daerah)", options=list(pemetaan_aliran.keys()))
    curah_hujan = st.number_input("Curah Hujan (mm)", min_value=0.0, step=0.1)
    debit_air = st.number_input("Debit Air (m³/s)", min_value=0.0, step=0.1)
    muka_air = st.number_input("Tinggi Muka Air (m)", min_value=0.0, step=0.1)
    # Tinggi banjir tetap ada sebagai input tetapi tidak digunakan fitur model XGB (sesuai ipynb)
    tinggi_banjir_input = st.number_input("Tinggi Genangan Air (m)", min_value=0.0, max_value=5.0, step=0.01)
    
    tombol_prediksi = st.button("🔍 Jalankan Prediksi", use_container_width=True, type="primary")

# --- FUNGSI LOAD DATA & TRAINING (ADAPTASI XGBOOST.IPYNB) ---
@st.cache_resource
def prepare_model(lokasi_terpilih):
    # Menggunakan file utama dari notebook
    filename = "Banjir all - Data Acak (1).csv"
    try:
        df_train = pd.read_csv(filename)
    except:
        return None, None, None, None

    # Cleaning sesuai notebook
    df_train = df_train.drop(columns=['Tanggal', 'Tinggi Banjir', 'Banjir Ya/Tidak'], errors='ignore')
    df_train = df_train.replace('-', np.nan)
    
    kolom_numerik = ['Curah Hujan', 'Debit Air', 'Muka Air']
    for col in kolom_numerik:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
    
    df_train = df_train.ffill().bfill()
    
    # Target Engineering (Level Banjir)
    df_train['Level_Banjir'] = df_train['Muka Air'].apply(tentukan_level_banjir)

    # Encoding Fitur (Kecamatan)
    le_kec = LabelEncoder()
    df_train['Kecamatan'] = le_kec.fit_transform(df_train['Kecamatan'])
    
    # Encoding Target
    le_target = LabelEncoder()
    df_train['Level_Banjir'] = le_target.fit_transform(df_train['Level_Banjir'])

    X = df_train.drop(columns=['Level_Banjir'])
    y = df_train['Level_Banjir']

    # Split untuk metrik
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model XGBoost sesuai notebook
    model_xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric='mlogloss'
    )
    model_xgb.fit(X_train, y_train)
    
    # Hitung metrik untuk tab Performa
    y_pred = model_xgb.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, target_names=le_target.classes_, output_dict=True),
        "gmean": geometric_mean_score(y_test, y_pred, average='macro'),
        "cm": confusion_matrix(y_test, y_pred)
    }

    return model_xgb, le_kec, le_target, metrics


DAFTAR_FILE_DATASET = [
    "Banjir all - Data Acak (1).csv",
    "Banjir_all_-_Data_Acak__1___1_.csv",
]


@st.cache_data
def load_dataset_mentah():
    """Memuat ulang CSV historis secara 'mentah' (kolom Tanggal & Kecamatan tetap utuh,
    tidak di-encode) khusus untuk dipakai sebagai acuan statistik/klimatologi pada tab
    Prakiraan 7 Hari. Mengembalikan None bila file tidak ditemukan, supaya tab tetap bisa
    jalan dalam mode simulasi murni (fallback)."""
    df_mentah = None
    for fname in DAFTAR_FILE_DATASET:
        try:
            df_mentah = pd.read_csv(fname)
            break
        except Exception:
            continue
    if df_mentah is None:
        return None

    df_mentah = df_mentah.replace('-', np.nan)
    for col in ['Curah Hujan', 'Debit Air', 'Muka Air']:
        if col in df_mentah.columns:
            df_mentah[col] = pd.to_numeric(df_mentah[col], errors='coerce')
    df_mentah['Tanggal'] = pd.to_datetime(df_mentah['Tanggal'], errors='coerce')
    df_mentah['DayOfYear'] = df_mentah['Tanggal'].dt.dayofyear
    df_mentah['Kecamatan'] = df_mentah['Kecamatan'].astype(str).str.strip()
    return df_mentah


def get_climatology(df_hist, stasiun, target_doy, window=7):
    """
    Mengambil rata-rata & standar deviasi historis Curah Hujan, Debit Air, dan TMA
    di sekitar tanggal target (+- `window` hari, lintas tahun 2020-2024) untuk stasiun
    tertentu — ini yang membuat prakiraan H+1..H+6 'berbasis dataset' alih-alih angka acak
    sembarangan. Fallback bertingkat dipakai bila data spesifik tidak cukup:
    1) stasiun terkait pada musim yang sama, 2) seluruh histori stasiun terkait,
    3) seluruh histori Dayeuhkolot (data terlengkap), 4) rata-rata seluruh dataset.
    """
    kolom = ['Curah Hujan', 'Debit Air', 'Muka Air']

    def ringkas(sub):
        if sub is None or sub.empty:
            return None
        if sub[kolom].dropna(how='all').empty:
            return None
        return {
            'hujan_mean': sub['Curah Hujan'].mean(skipna=True),
            'hujan_std': sub['Curah Hujan'].std(skipna=True),
            'debit_mean': sub['Debit Air'].mean(skipna=True),
            'debit_std': sub['Debit Air'].std(skipna=True),
            'muka_mean': sub['Muka Air'].mean(skipna=True),
            'muka_std': sub['Muka Air'].std(skipna=True),
            'n': len(sub),
        }

    sub_stasiun = df_hist[df_hist['Kecamatan'] == stasiun]

    diff = (sub_stasiun['DayOfYear'] - target_doy).abs()
    diff = np.minimum(diff, 365 - diff)
    musiman = ringkas(sub_stasiun[diff <= window]) if not sub_stasiun.empty else None
    if musiman and pd.notna(musiman['muka_mean']) and musiman['n'] >= 3:
        hasil = musiman
    else:
        hasil = ringkas(sub_stasiun)

    fallback_dayeuhkolot = ringkas(df_hist[df_hist['Kecamatan'] == 'Dayeuhkolot'])
    fallback_global = ringkas(df_hist)

    if hasil is None:
        hasil = fallback_dayeuhkolot or fallback_global or {}

    # Tambal field per-field yang masih NaN (mis. stasiun tidak punya data Debit/TMA)
    for sumber in (fallback_dayeuhkolot, fallback_global):
        if sumber is None:
            continue
        for k in ['hujan_mean', 'hujan_std', 'debit_mean', 'debit_std', 'muka_mean', 'muka_std']:
            if k not in hasil or pd.isna(hasil.get(k)):
                hasil[k] = sumber.get(k)

    # Penjaga terakhir bila tetap NaN semua
    default_aman = {
        'hujan_mean': 10.0, 'hujan_std': 12.0,
        'debit_mean': 40.0, 'debit_std': 20.0,
        'muka_mean': 0.6, 'muka_std': 0.3,
    }
    for k, v in default_aman.items():
        if k not in hasil or pd.isna(hasil.get(k)):
            hasil[k] = v

    return hasil


def generate_weekly_forecast(model, le_kec, le_target, df_hist, aliran_induk,
                              base_hujan, base_debit, base_muka, seed=None):
    """
    Membuat prakiraan 7 hari ke depan (Hari Ini s.d. H+6).

    Hari ke-0 (Hari Ini) memakai nilai input sidebar apa adanya.
    Untuk H+1..H+6, dipakai pendekatan 'persistence -> klimatologi':
    nilai hari sebelumnya ditarik berangsur-angsur menuju rata-rata historis
    (dari dataset CSV asli) untuk tanggal yang sama di tahun-tahun sebelumnya
    pada stasiun terkait, lalu ditambah noise acak yang skalanya juga diambil
    dari standar deviasi historis stasiun tersebut — bukan angka acak sembarangan.

    Bila dataset historis tidak tersedia (df_hist None), fungsi otomatis jatuh
    kembali ke random-walk sederhana seperti sebelumnya.
    """
    rng = np.random.default_rng(seed)

    daftar_kelas = list(le_kec.classes_)
    stasiun_cocok = cocokkan_kecamatan(daftar_kelas, aliran_induk)
    try:
        kec_encoded = le_kec.transform([stasiun_cocok])[0]
    except Exception:
        kec_encoded = 0

    today_doy = datetime.now().timetuple().tm_yday

    hujan_prev, debit_prev, muka_prev = base_hujan, base_debit, base_muka
    hasil = []
    klimatologi_dipakai = df_hist is not None

    for i in range(7):
        if i == 0:
            hujan, debit, muka = base_hujan, base_debit, base_muka
        else:
            if klimatologi_dipakai:
                target_doy = ((today_doy - 1 + i) % 365) + 1
                clim = get_climatology(df_hist, stasiun_cocok, target_doy)

                # Bobot persistence menurun seiring horizon (H+1 masih dekat kondisi
                # hari ini, H+6 makin condong ke pola musiman historis)
                w_persist = max(0.15, 0.70 - 0.10 * i)

                target_hujan = w_persist * hujan_prev + (1 - w_persist) * clim['hujan_mean']
                target_debit = w_persist * debit_prev + (1 - w_persist) * clim['debit_mean']
                target_muka = w_persist * muka_prev + (1 - w_persist) * clim['muka_mean']

                noise_hujan = rng.normal(0, max(clim['hujan_std'], 4.0) * 0.4)
                noise_debit = rng.normal(0, max(clim['debit_std'], 6.0) * 0.4)
                noise_muka = rng.normal(0, max(clim['muka_std'], 0.08) * 0.4)

                hujan = float(np.clip(target_hujan + noise_hujan, 0, 250))
                debit = float(np.clip(target_debit + noise_debit, 0, 700))
                muka = float(np.clip(target_muka + noise_muka, 0, 10))
            else:
                hujan = float(np.clip(hujan_prev + rng.normal(0, 18), 0, 200))
                debit = float(np.clip(debit_prev + rng.normal(0, 25), 0, 350))
                muka = float(np.clip(muka_prev + rng.normal(0, 0.18), 0, 3.5))

        input_df = pd.DataFrame(
            [[kec_encoded, hujan, debit, muka]],
            columns=['Kecamatan', 'Curah Hujan', 'Debit Air', 'Muka Air']
        )
        idx_pred = model.predict(input_df)[0]
        label_pred = le_target.inverse_transform([idx_pred])[0]
        conf = float(max(model.predict_proba(input_df)[0]))

        hasil.append({
            "hujan": hujan,
            "debit": debit,
            "muka": muka,
            "label": label_pred,
            "confidence": conf
        })
        hujan_prev, debit_prev, muka_prev = hujan, debit, muka

    return hasil, stasiun_cocok, klimatologi_dipakai


# Inisialisasi model
model, le_kec, le_target, model_metrics = prepare_model(lokasi_select)
df_hist = load_dataset_mentah()

# --- MAIN UI ---
st.title(" Sistem Peringatan Dini Banjir Berbasis Aliran Sungai")
st.markdown("Pantau dan prediksi potensi banjir di wilayah Kabupaten Bandung menggunakan algoritma **XGBoost Classifier**.")

if model is None:
    st.error("File 'Banjir all - Data Acak (1).csv' tidak ditemukan!")
    st.stop()

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "Prediksi Level Siaga & Peta",
    " Simulasi Real-time",
    "Performa Model AI",
    "📅 Prakiraan 7 Hari"
])

with tab1:
    col_hasil, col_peta = st.columns([1, 1.2]) 
    
    with col_hasil:
        st.subheader("Hasil Analisis Level Banjir")
        if tombol_prediksi:
            # Preprocessing input
            aliran_induk = pemetaan_aliran.get(lokasi_select, "Dayeuhkolot")
            # Encode lokasi (cocokkan nama aliran induk ke nama Kecamatan asli di dataset)
            stasiun_cocok_t1 = cocokkan_kecamatan(list(le_kec.classes_), aliran_induk)
            try:
                kec_encoded = le_kec.transform([stasiun_cocok_t1])[0]
            except:
                kec_encoded = 0 # Default ke index pertama jika unknown
            
            input_df = pd.DataFrame([[kec_encoded, curah_hujan, debit_air, muka_air]], 
                                   columns=['Kecamatan', 'Curah Hujan', 'Debit Air', 'Muka Air'])
            
            # Prediksi
            res_idx = model.predict(input_df)[0]
            res_label = le_target.inverse_transform([res_idx])[0]
            probs = model.predict_proba(input_df)[0]
            confidence = max(probs)

            st.info(f"ℹ️ Analisis berdasarkan stasiun utama: **{aliran_induk}**")
            if stasiun_cocok_t1 != aliran_induk:
                st.caption(f"Dicocokkan ke data historis stasiun: *{stasiun_cocok_t1}* (nama 'aliran induk' tidak persis sama dengan nama Kecamatan di dataset).")
            
            if "Awas" in res_label:
                st.error(f"🚨 **STATUS: {res_label}**")
                st.write("Segera lakukan evakuasi dan amankan barang berharga!")
            elif "Siaga" in res_label:
                st.warning(f"⚠️ **STATUS: {res_label}**")
                st.write("Waspada, air mulai memasuki pemukiman.")
            elif "Waspada" in res_label:
                st.warning(f"🟡 **STATUS: {res_label}**")
                st.write("Siaga terhadap kenaikan debit air kiriman.")
            else:
                st.success(f"✅ **STATUS: {res_label}**")
                st.write("Kondisi saat ini terpantau aman.")
                
            st.progress(float(confidence), text=f"Tingkat Keyakinan Model: {confidence:.2%}")
            
            st.write("---")
            st.write("**Data Input:**")
            st.write(f"- TMA: {muka_air} m | Curah Hujan: {curah_hujan} mm")
        else:
            st.info("👈 Silakan atur parameter di panel samping dan tekan tombol 'Jalankan Prediksi'.")

    with col_peta:
        st.subheader("Peta Pantauan Sungai (GIS)")
        stasiun_utama = pemetaan_aliran.get(lokasi_select, "Dayeuhkolot")
        
        if HAS_FOLIUM:
            koor = koordinat_stasiun.get(stasiun_utama, [-6.9881, 107.6281]) 
            m = folium.Map(location=koor, zoom_start=13, tiles="CartoDB positron")
            
            folium.Marker(
                koor, 
                popup=f"Stasiun Acuan: {stasiun_utama}", 
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
            st.warning("Library 'folium' belum terinstal.")

with tab2:
    st.subheader("Pantauan Sensor Virtual (Simulasi Real-time)")
    
    if st.button("Cek Kondisi Terkini (Simulasi)"):
        # Random data simulasi
        sim_hujan = random.uniform(0, 120)
        sim_debit = random.uniform(20, 200)
        sim_muka = random.uniform(0.1, 1.8)
        
        # Prediksi simulasi
        sim_input = pd.DataFrame([[0, sim_hujan, sim_debit, sim_muka]], 
                                columns=['Kecamatan', 'Curah Hujan', 'Debit Air', 'Muka Air'])
        sim_idx = model.predict(sim_input)[0]
        sim_label = le_target.inverse_transform([sim_idx])[0]
        
        wib_now = datetime.utcnow() + timedelta(hours=7)
        
        # UI Box Status
        bg_color = "#ffebeb" if "Awas" in sim_label or "Siaga" in sim_label else "#e8fdf0"
        border_color = "red" if "Awas" in sim_label or "Siaga" in sim_label else "green"
        
        st.markdown(f"""
        <div style="padding: 15px; border-radius: 10px; background-color: {bg_color}; border: 1px solid {border_color};">
            <h3>📢 Status: {sim_label}</h3>
            <p>Diperbarui pada: <b>{wib_now.strftime("%H:%M:%S WIB")}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        k1, k2, k3 = st.columns(3)
        k1.metric("Curah Hujan", f"{sim_hujan:.1f} mm")
        k2.metric("Debit Air", f"{sim_debit:.1f} m³/s")
        k3.metric("Muka Air (TMA)", f"{sim_muka:.2f} m")

with tab3:
    st.subheader("Detail Evaluasi Algoritma XGBoost")
    
    m1, m2 = st.columns(2)
    m1.metric("Akurasi Model", f"{model_metrics['accuracy']:.2%}")
    m2.metric("G-Mean Score", f"{model_metrics['gmean']:.4f}")
    
    st.divider()
    col_cm, col_rep = st.columns([1, 1.5])
    
    with col_cm:
        st.write("**Confusion Matrix:**")
        cm_df = pd.DataFrame(
            model_metrics['cm'], 
            index=[f"Aktual {c}" for c in le_target.classes_],
            columns=[f"Prediksi {c}" for c in le_target.classes_]
        )
        st.table(cm_df)
        
    with col_rep:
        st.write("**Detail Laporan Klasifikasi per Level:**")
        report_df = pd.DataFrame(model_metrics['report']).transpose()
        st.dataframe(report_df.style.format(precision=2))

    st.info("""
    **Catatan Teknis:**
    - Model menggunakan **XGBoost Classifier** dengan parameter `mlogloss`.
    - Klasifikasi dibagi menjadi 4 kelas sesuai standar TMA di notebook.
    - Data dilatih menggunakan dataset: `Banjir all - Data Acak (1).csv`.
    """)

with tab4:
    st.subheader("📅 Prakiraan Potensi Banjir 7 Hari ke Depan")
    aliran_induk_fc = pemetaan_aliran.get(lokasi_select, "Dayeuhkolot")
    st.write(
        f"Prakiraan disusun untuk stasiun acuan **{aliran_induk_fc}**, menggunakan kondisi "
        f"Curah Hujan, Debit Air, dan TMA saat ini sebagai titik awal tren, lalu diproyeksikan "
        f"6 hari ke depan dan dinilai ulang oleh model XGBoost yang sama."
    )

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        buat_prakiraan = st.button("🔄 Buat / Perbarui Prakiraan 7 Hari", use_container_width=True)

    perlu_generate = (
        buat_prakiraan
        or "weekly_forecast" not in st.session_state
        or st.session_state.get("weekly_forecast_lokasi") != lokasi_select
    )

    if perlu_generate:
        base_hujan = curah_hujan if curah_hujan > 0 else random.uniform(5, 40)
        base_debit = debit_air if debit_air > 0 else random.uniform(30, 100)
        base_muka = muka_air if muka_air > 0 else random.uniform(0.2, 0.8)

        forecast_baru, stasiun_acuan_fc, pakai_klimatologi = generate_weekly_forecast(
            model, le_kec, le_target, df_hist, aliran_induk_fc,
            base_hujan, base_debit, base_muka,
            seed=random.randint(0, 99999)
        )
        st.session_state["weekly_forecast"] = forecast_baru
        st.session_state["weekly_forecast_stasiun"] = stasiun_acuan_fc
        st.session_state["weekly_forecast_klimatologi"] = pakai_klimatologi
        st.session_state["weekly_forecast_lokasi"] = lokasi_select
        st.session_state["weekly_forecast_time"] = datetime.utcnow() + timedelta(hours=7)

    forecast = st.session_state["weekly_forecast"]
    waktu_buat = st.session_state.get("weekly_forecast_time")
    stasiun_acuan_fc = st.session_state.get("weekly_forecast_stasiun", aliran_induk_fc)
    pakai_klimatologi = st.session_state.get("weekly_forecast_klimatologi", False)

    if pakai_klimatologi:
        st.success(
            f"✅ Prakiraan H+1 s.d. H+6 memakai rata-rata & variasi historis 2020-2024 "
            f"(klimatologi) dari stasiun **{stasiun_acuan_fc}** pada dataset, dipadukan dengan "
            f"kondisi hari ini sebagai titik awal."
        )
    else:
        st.warning(
            "⚠️ Dataset historis (`Banjir all - Data Acak (1).csv`) tidak ditemukan di server. "
            "Prakiraan H+1 s.d. H+6 sementara memakai simulasi random-walk sederhana, belum "
            "berbasis data historis."
        )

    # --- Label hari dalam Bahasa Indonesia ---
    hari_full_id = {0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'}
    bulan_id = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'Mei', 6: 'Jun',
                7: 'Jul', 8: 'Agu', 9: 'Sep', 10: 'Okt', 11: 'Nov', 12: 'Des'}
    today = datetime.now()
    labels_hari = []
    for i in range(7):
        d = today + timedelta(days=i)
        if i == 0:
            labels_hari.append("Hari Ini")
        elif i == 1:
            labels_hari.append("Besok")
        else:
            labels_hari.append(hari_full_id[d.weekday()])
    tanggal_hari_ini = f"{hari_full_id[today.weekday()]}, {today.day} {bulan_id[today.month]} {today.year}"

    # --- Ringkasan "Hari Ini" ala widget cuaca ---
    style_today = get_level_style(forecast[0]["label"])
    st.markdown(f"""
    <div style="background:#14161c; border-radius:16px; padding:24px 28px;
                display:flex; justify-content:space-between; align-items:center;
                flex-wrap:wrap; gap:12px; margin-bottom:18px;">
        <div style="display:flex; align-items:center; gap:18px;">
            <div style="font-size:56px; line-height:1;">{style_today['icon']}</div>
            <div>
                <div style="color:#fff; font-size:36px; font-weight:700; line-height:1.1;">{forecast[0]['muka']:.2f} m</div>
                <div style="color:#9aa0a6; font-size:13px; margin-top:6px;">
                    Curah Hujan: {forecast[0]['hujan']:.0f} mm &nbsp;|&nbsp;
                    Debit Air: {forecast[0]['debit']:.0f} m³/s &nbsp;|&nbsp;
                    Keyakinan: {forecast[0]['confidence']:.0%}
                </div>
            </div>
        </div>
        <div style="text-align:right;">
            <div style="color:#fff; font-size:20px; font-weight:600;">Prakiraan Banjir</div>
            <div style="color:#9aa0a6; font-size:13px;">{tanggal_hari_ini}</div>
            <div style="color:{style_today['color']}; font-size:16px; font-weight:700;">{style_today['short']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Grafik tren TMA 7 hari (gaya area chart seperti widget cuaca) ---
    tma_values = [h["muka"] for h in forecast]
    fig, ax = plt.subplots(figsize=(10, 2.6))
    fig.patch.set_facecolor('#14161c')
    ax.set_facecolor('#14161c')
    x = list(range(7))
    ax.plot(x, tma_values, color='#f5c518', linewidth=2.5, zorder=3)
    ax.fill_between(x, tma_values, color='#6b6b1a', alpha=0.55, zorder=2)
    y_pad = (max(tma_values) - min(tma_values)) * 0.15 + 0.05
    for i, v in enumerate(tma_values):
        ax.text(i, v + y_pad * 0.35, f"{v:.2f}", color='white', fontsize=9, ha='center')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_hari, color='#9aa0a6', fontsize=9)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='x', colors='#9aa0a6', length=0)
    ax.margins(y=0.3)
    st.pyplot(fig, use_container_width=True)

    st.write("")

    # --- Kartu per hari (mirip kartu Fri/Sat/Sun... pada referensi) ---
    cols = st.columns(7)
    for i, (col, hari) in enumerate(zip(cols, forecast)):
        style = get_level_style(hari["label"])
        border_w = "2px" if i == 0 else "1px"
        with col:
            st.markdown(f"""
            <div style="background:{style['bg']}; border-radius:14px; padding:14px 6px;
                        text-align:center; border:{border_w} solid {style['color']}66;">
                <div style="color:#cfd2d6; font-size:12px; margin-bottom:6px;">{labels_hari[i]}</div>
                <div style="font-size:30px;">{style['icon']}</div>
                <div style="color:{style['color']}; font-weight:700; font-size:12px; margin-top:6px;">{style['short']}</div>
                <div style="color:#fff; font-size:13px; margin-top:4px;">{hari['muka']:.2f} m</div>
                <div style="color:#9aa0a6; font-size:11px; margin-top:2px;">{hari['hujan']:.0f} mm</div>
            </div>
            """, unsafe_allow_html=True)

    if waktu_buat:
        st.caption(
            f"Prakiraan dibuat pada {waktu_buat.strftime('%H:%M:%S WIB')}. "
            "Catatan: H+1 s.d. H+6 disimulasikan secara statistik dari kondisi hari ini "
            "(belum terhubung ke data prakiraan cuaca real-time seperti BMKG)."
        )

    st.divider()
    st.write("**Keterangan Level:**")
    leg_cols = st.columns(4)
    legend_items = [
        ("☀️", "#4dd07a", "0 - Normal", "TMA < 0.57 m"),
        ("⛅", "#f5d547", "1 - Waspada", "0.57 - 0.93 m"),
        ("🌧️", "#ff9f43", "2 - Siaga", "0.93 - 1.30 m"),
        ("⛈️", "#ff5c5c", "3 - Awas", "> 1.30 m"),
    ]
    for col, (icon, color, label, rng_txt) in zip(leg_cols, legend_items):
        with col:
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="font-size:22px;">{icon}</div>
                <div style="color:{color}; font-weight:600; font-size:13px;">{label}</div>
                <div style="color:#9aa0a6; font-size:11px;">{rng_txt}</div>
            </div>
            """, unsafe_allow_html=True)
