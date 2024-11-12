import streamlit as st
import pandas as pd
from xgboost import XGBClassifier

# Title of the app
st.title("Prediksi Kelayakan Pinjaman")

# Load the model
loaded_model = XGBClassifier()
try:
    loaded_model.load_model('xgb_model.json')  # Update with the correct path if necessary
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Collect user input
Age = st.number_input("Usia", min_value=18, max_value=100)

Experience = st.number_input("Jumlah Tahun Pengalaman Kerja", min_value=1, max_value=100)

Family = st.number_input("Jumlah Keluarga", min_value=1, max_value=50)

Education = st.selectbox("Pendidikan Terakhir", options=["1", "2", "3"])
st.caption(":red[**1**]: Sarjana (Bachelor's degree) :red[**2**]: Magister (Master's degree) :red[**3**]: Profesional (Professional degree)")

Securities_Account = st.selectbox("Apakah memiliki Rekening Investasi?", options=[0, 1])
st.caption(":red[**0**] :  Tidak Ada;  :red[**1**] :  Ada")

CD_Account = st.selectbox("Apakah memiliki Rekening CD (Sertifikat Deposito)?", options=[0, 1])
st.caption(":red[**0**] :  Tidak Ada;  :red[**1**] :  Ada")

Online = st.selectbox("Apakah menggunakan layanan Mobile Banking?", options=[0, 1])
st.caption(":red[**0**] :  Tidak Ada;  :red[**1**] :  Ada")

CreditCard = st.selectbox("Apakah memiliki Kartu Kredit?", options=[0, 1])
st.caption(":red[**0**] :  Tidak Ada;  :red[**1**] :  Ada")

Income = st.number_input("Jumlah Pendapatan Bulanan (dalam Rupiah)", min_value=0, max_value=500000000)

CCAvg = st.number_input("Jumlah Pengeluaran Bulanan (dalam Rupiah)", min_value=0, max_value=100000000)

Mortgage = st.number_input("Jumlah Hipotek (dalam Rupiah)", min_value=0, max_value=100000000)
st.caption("Nilai barang yang dijadikan jaminan nasabah kepada bank")

# Create a DataFrame from user input
input_data = pd.DataFrame({
    'Age': [Age],
    'Experience': [Experience],
    'Income': [Income],
    'Family': [Family],
    'CCAvg': [CCAvg],
    'Education': [int(Education)],
    'Mortgage': [Mortgage],
    'Securities Account': [Securities_Account],
    'CD Account': [CD_Account],
    'Online': [Online],
    'CreditCard': [CreditCard],
})

st.write(input_data)  # For debugging

# Ensure all data types are numeric and there are no missing values
input_data = input_data.astype(float).fillna(0)

# Make prediction
if st.button("Submit"):
    try:
        prediction = loaded_model.predict(input_data)
        prediction_proba = loaded_model.predict_proba(input_data)

        if prediction[0] == 1:
            st.success(f"Nasabah layak diberikan pinjaman dengan nilai prediksi {prediction_proba[0][1] * 100:.2f}%")
        else:
            st.error(f"Nasabah belum layak diberikan pinjaman dengan nilai prediksi {prediction_proba[0][0] * 100:.2f}%")
    except ValueError as e:
        st.error(f"Terjadi kesalahan pada input data: {e}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
