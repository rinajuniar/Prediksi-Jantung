import streamlit as st
from web_functions import predict

def app(df, x, y):
    st.title("Halaman Prediksi Penyakit Jantung")

    sex         = st.number_input ('Masukan Nilai sex')
    cp          = st.number_input ('Masukan Nilai cp')
    fbs         = st.number_input ('Masukan Nilai fbs')
    restecg     = st.number_input ('Masukan Nilai restecg')
    exng        = st.number_input ('Masukan Nilai exng')
    slp         = st.number_input ('Masukan Nilai slp')
    caa         = st.number_input ('Masukan Nilai caa')
    thall       = st.number_input ('Masukan Nilai thall')

    features = [sex,cp,fbs,restecg,exng,slp,caa,thall]

#tombol prediksi
    if st.button("Prediksi"):
        prediction, score= predict(x, y, features)
        score = score
        st.info("Prediksi Sukses!!!!")
        
        if prediction is not None:
            if(prediction==1):
             st.warning("Pasien Terkena Penyakit Jantung.")
            else:
                st.success("Pasien Tidak Terkena Penyakit Jantung.")
        st.write("Model Yang Digunakan Memiliki Tingkat Akurasi",(score*100),"%")