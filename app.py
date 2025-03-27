from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import streamlit as st
import pandas as pd
import numpy as np
import pickle


#modeli yükleme

modela=pickle.load(open("ridge_test.pickle","rb"))

scaler=pickle.load(open("scaler.pickle","rb"))



# Başlık
st.title("Makine Öğrenmesi ile Tahmin")
st.write("Girdiğiniz değerlere göre tahminleme yapabilirsiniz.")

# Kullanıcıdan giriş alma

gunluk_calisma_saati = st.sidebar.slider("Günlük çalışma saati", min_value=0, max_value=24, value=12)

onceki_puan = st.sidebar.slider("Önceki puan", min_value=0, max_value=100, value=50)

uyku_saati = st.sidebar.slider("Uyku saati", min_value=0, max_value=24, value=12)

cozulen_ornek_sayisi = st.sidebar.slider("Çözülen örnek sayısı", min_value=0, max_value=50, value=0)

ders_disi_etkinlik_dict = {'Hayır': 0, 'Evet': 1}
ders_disi_etkinlik = ders_disi_etkinlik_dict[st.sidebar.selectbox("Ders Dışı Etkinlik", ders_disi_etkinlik_dict.keys())]

# Özellikleri bir dataframe'e çevirme ve standartlaştırma
yeni_ogr = scaler.transform(np.array([gunluk_calisma_saati, onceki_puan, ders_disi_etkinlik, uyku_saati, cozulen_ornek_sayisi]).reshape(1, -1))



# Tahmin butonu
if st.button("Tahmin Yap"):
    prediction = modela.predict(yeni_ogr)
    prediction = min(prediction[0], 100)  # Tahmin sonucu 100'ü geçerse 100 olarak sınırla
    st.write(f"Tahmin Sonucu: {prediction:.2f}")

#MODEL KATSAYILARI TABLOSU
st.subheader("Modelin Katsayıları")

# Özellik isimleri
features = ["Günlük Çalışma Saati", "Önceki Puan", "Ders Dışı Etkinlik", "Uyku Saati", "Çözülen Örnek Sayısı"]
coefficients = modela.coef_

# DataFrame oluştur ve göster
coef_df = pd.DataFrame({"Özellik": features, "Katsayı": coefficients})
st.table(coef_df.style.format({"Katsayı": "{:.4f}"}))
