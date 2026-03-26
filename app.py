import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Prediksi Banjir Dayeuhkolot", layout="wide")

# ================================
# MAPPING
# ================================
pemetaan_aliran = {
    "Dayeuhkolot": "Dayeuhkolot",
    "Situ Cisanti (Kertasari)": "Dayeuhkolot",
    "Wangisagara (Majalaya)": "Dayeuhkolot",
    "Sapan": "Dayeuhkolot",
    "Rancamanyar": "Dayeuhkolot",
    "Nanjung": "Dayeuhkolot",
    "Hantap": "Dayeuhkolot",

    "Cipanas - Margamukti": "Cipanas - Margamukti",
    "Cileunca": "Cipanas - Margamukti",
    "Kertamanah": "Cipanas - Margamukti",

    "Cikeruh - Jatiroke": "Cikeruh - Jatiroke",
    "Cicalengka": "Cikeruh - Jatiroke",
    "Ciluluk": "Cikeruh - Jatiroke",

    "Cisondari - Pasirjambu": "Cisondari - Pasirjambu",
    "Ciwidey": "Cisondari - Pasirjambu",

    "Bojongsoang": "Bojongsoang"
}

# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_data():
    file_mapping = {
        "Dayeuhkolot": "Data Banjir Daleuhlkolot - Sheet1.csv",
        "Bojongsoang": "Bojongsoang (2020-2024).xlsx"
    }

    dfs = []

    for stasiun, file in file_mapping.items():
        try:
            if file.endswith(".xlsx"):
                df = pd.read_excel(file)
            else:
                df = pd.read_csv(file, encoding='utf-8', encoding_errors='replace')

            df.columns = df.columns.str.strip()

            # cari kolom banjir
            for col in df.columns:
                if "banjir" in col.lower():
                    df.rename(columns={col: "Banjir Ya/Tidak"}, inplace=True)

            df["Kecamatan"] = stasiun
            dfs.append(df)

        except:
            pass

    df = pd.concat(dfs, ignore_index=True)

    # target cleaning
    df["Banjir Ya/Tidak"] = df["Banjir Ya/Tidak"].astype(str).str.lower()
    df["Banjir Ya/Tidak"] = df["Banjir Ya/Tidak"].map({"ya":1,"tidak":0,"1":1,"0":0})
    df = df.dropna(subset=["Banjir Ya/Tidak"])

    # numerik
    cols = ["Curah Hujan","Debit Air","Muka Air","Tinggi Banjir"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # encoding kecamatan
    unique_kec = sorted(df["Kecamatan"].unique())
    mapping = {k:i for i,k in enumerate(unique_kec)}
    df["Kecamatan_Enc"] = df["Kecamatan"].map(mapping)

    return df, mapping

# ================================
# BALANCING 1:3
# ================================
def balance_data(df, ratio=3):
    df_banjir = df[df["Banjir Ya/Tidak"] == 1]
    df_tidak = df[df["Banjir Ya/Tidak"] == 0]

    n_banjir = len(df_banjir)
    target_tidak = ratio * n_banjir

    # undersample atau oversample tidak banjir
    if len(df_tidak) > target_tidak:
        df_tidak = df_tidak.sample(target_tidak, random_state=42)
    else:
        df_tidak = df_tidak.sample(target_tidak, replace=True, random_state=42)

    df_final = pd.concat([df_banjir, df_tidak])
    return df_final.sample(frac=1, random_state=42)

# ================================
# TRAIN MODEL
# ================================
@st.cache_resource
def train(df):
    features = ["Kecamatan_Enc","Curah Hujan","Debit Air","Muka Air","Tinggi Banjir"]

    X = df[features]
    y = df["Banjir Ya/Tidak"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,y,test_size=0.2,stratify=y,random_state=42
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return (
        model,
        features,
        accuracy_score(y_test,y_pred),
        precision_score(y_test,y_pred),
        recall_score(y_test,y_pred),
        f1_score(y_test,y_pred),
        confusion_matrix(y_test,y_pred),
        classification_report(y_test,y_pred, output_dict=True)
    )

# ================================
# MAIN FLOW
# ================================
df, mapping = load_data()

# 🔥 BALANCING DI SINI
df = balance_data(df, ratio=3)

model, features, acc, prec, rec, f1, cm, report = train(df)

# ================================
# UI
# ================================
st.title("🚨 Prediksi Banjir AI")

# sidebar input
lokasi = st.sidebar.selectbox("Lokasi", list(pemetaan_aliran.keys()))
hujan = st.sidebar.number_input("Curah Hujan",0.0)
debit = st.sidebar.number_input("Debit Air",0.0)
muka = st.sidebar.number_input("Muka Air",0.0)
tinggi = st.sidebar.number_input("Tinggi Banjir",0.0)

if st.sidebar.button("Prediksi"):
    lokasi_utama = pemetaan_aliran[lokasi]
    kode = mapping.get(lokasi_utama,0)

    input_df = pd.DataFrame([[kode,hujan,debit,muka,tinggi]],columns=features)

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"⚠️ BANJIR ({prob:.2%})")
    else:
        st.success(f"✅ AMAN ({prob:.2%})")

# ================================
# METRICS
# ================================
st.subheader("Performa Model")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Accuracy", f"{acc:.2%}")
c2.metric("Precision", f"{prec:.2%}")
c3.metric("Recall (Banjir)", f"{rec:.2%}")
c4.metric("F1 Score", f"{f1:.2%}")

st.write("Confusion Matrix")
st.write(cm)

st.write("Distribusi Data Setelah Balancing")
st.write(df["Banjir Ya/Tidak"].value_counts())
