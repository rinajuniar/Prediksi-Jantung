# Laporan Proyek Machine Learning
### Nama : Rina Juniar
### Nim : 211351126
### Kelas : Pagi B

## Domain Proyek

Dapat digunakan sebagai sebuah sistem aplikasi untuk memprediksi apakah seseorang terkena penyakit serangan jantung
## Business Understanding

Mengembangkan model prediksi serangan penyakit jantung untuk membantu identifikasi risiko potensial pada pasien, memungkinkan intervensi dini dan perawatan yang tepat waktu.

Bagian laporan ini mencakup:

### Problem Statements

- Keterbatasan Sistem Prediksi Heart Attack yang Ada:
Saat ini, terdapat keterbatasan dalam sistem prediksi serangan jantung yang tersedia. Beberapa model prediksi belum sepenuhnya akurat dan sensitif, menyebabkan tingkat kepercayaan yang rendah pada hasil prediksi tersebut.

-Kurangnya Integrasi Data yang Komprehensif:
Kurangnya integrasi data yang komprehensif menjadi hambatan dalam mengembangkan sistem prediksi serangan jantung yang efektif. Informasi medis dari berbagai sumber seringkali tidak terpadu dengan baik, mengakibatkan kurangnya kelengkapan data.

- Ketidakmampuan Mengidentifikasi Faktor Risiko yang Spesifik:
Beberapa sistem prediksi saat ini mungkin belum mampu mengidentifikasi faktor risiko yang spesifik dengan akurat. Hal ini dapat mengakibatkan penyelewengan dalam penilaian risiko individu, mengurangi efektivitas prediksi serangan jantung.

- Keterbatasan Teknologi yang Digunakan:
Adanya keterbatasan dalam teknologi yang digunakan untuk pengembangan sistem prediksi serangan jantung dapat membatasi kemampuan sistem untuk melakukan analisis mendalam dan akurat terhadap data medis kompleks.

- Kesulitan dalam Penyesuaian Model pada Populasi Beragam:
Setiap individu memiliki karakteristik kesehatan yang unik, dan pengembangan model prediksi serangan jantung yang dapat menyesuaikan diri dengan populasi yang beragam menjadi tantangan. Kesulitan ini dapat memengaruhi tingkat akurasi model di berbagai kelompok populasi.

- Kurangnya Edukasi dan Kesadaran Masyarakat:
Kurangnya edukasi dan kesadaran masyarakat mengenai pentingnya prediksi dini serangan jantung dapat menghambat penerimaan dan adopsi sistem prediksi. Masyarakat perlu didorong untuk lebih memahami manfaatnya dan mengakses sistem prediksi dengan lebih aktif.

- Tantangan dalam Pengumpulan Data Real-Time:
Pengumpulan data medis secara real-time menjadi tantangan dalam membangun sistem prediksi yang responsif. Keterlambatan dalam pengumpulan informasi dapat mengurangi kemampuan sistem untuk memberikan peringatan dini dengan tepat waktu.

- Ketidakpastian Terkait Faktor-faktor Lingkungan:
Faktor-faktor lingkungan seperti pola makan, aktivitas fisik, dan stres dapat mempengaruhi risiko serangan jantung. Pengidentifikasian dan penanganan faktor-faktor lingkungan ini dalam sistem prediksi serangan jantung masih menjadi tantangan tersendiri.

### Goals

- mencari solusi untuk memudahkan pasien atau seseorang mengetahui dirinya kemungkinan terkena penyakit serangan jantung atau tidak

    ### Solution statements
    - pengebangan platform sistem informasi tentang prediksi penyakit serangan jantung
    - Model yang dihasilkan dari datasets itu menggunakan metode KNN.

## Data Understanding
Dataset yang saya gunakan berasal jadi Kaggle yang berisi data prediksi diabetes.Dataset ini mengandung 304 baris dan 14 columns.

https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset 

### Variabel-variabel sebagai berikut:
- Usia : Usia pasien

- Jenis Kelamin : Jenis kelamin pasien

- exang: angina akibat olahraga (1 = ya; 0 = tidak)

- ca: jumlah kapal besar (0-3)

- cp : Tipe nyeri dada Tipe nyeri dada

Nilai 1 : angina tipikal
Nilai 2: angina atipikal
Nilai 3: nyeri non-angina
Nilai 4: tanpa gejala
trtbps : tekanan darah istirahat (dalam mm Hg)

- chol : kolestoral dalam mg/dl diambil melalui sensor BMI

- fbs : (gula darah puasa >120 mg/dl) (1 = benar; 0 = salah)

- rest_ecg : hasil elektrokardiografi istirahat

Nilai 0: biasa
Nilai 1 : mengalami kelainan gelombang ST-T (inversi gelombang T dan/atau elevasi atau depresi ST > 0,05 mV)
Nilai 2: menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri menurut kriteria Estes
thalach : tercapai denyut jantung maksimal

- target : 0= lebih kecil kemungkinan terkena serangan jantung 1= lebih besar kemungkinan terkena serangan jantung

## Data Preparation

DESKRIPSI LIBRARY
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import accuracy_score

```
MEMANGGIL DATASET
```python
df = pd.read_csv('diabetes.csv')
```
DESKRIPSI DATASET
```python
df.head()
```
```python
df.info()
```
```python
sns.heatmap(df.isnull())
```
![image](image1.png)
```pyyhon
Glucose = df.groupby('Age').count()[['BMI']].sort_values(by='BMI').reset_index()
Glucose = Glucose.rename(columns={'BMI' :'count'})
```
```python
plt.figure(figsize=(15,5))
sns.barplot(x=Glucose['Age'],y=Glucose['count'],color='purple')
```
![image](image2.png)
```
PISAHKAN DATA ATRIBUT DENGAN LABEL

```python
X = diabetes_data.drop(columns = 'Outcome', axis=1)
Y = diabetes_data['Outcome']
```
```python
print(X)
```

```python
print(Y)
```
PISAHKAN DATA TRAINING DAN DATA TESTING
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2 )
```
```python
print(X.shape, X_train.shape, X_test.shape)
```
MEMBUAT MODEL TRAINING
```python
model = LogisticRegression()
```
```python
model.fit(X_train, Y_train)
```
EVALUASI MODEL 
```X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
```
```pyhton
print('akurasi data training :', training_data_accuracy)
```
```python
X_test_prediction = model.predict(X_test)
test_data_accuracy =accuracy_score(X_test_prediction, Y_test)
```
```python
print('akurasi data testing :', test_data_accuracy)
```
```python
input_data = (6, 148, 72, 35, 0, 33.6, 0.267, 50)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
  print('Pasien Tidak Terkena Penyakit Diabetes')
else:
  print('Pasien Terkena Penyakit Diabetes')
```
SIMPAN MODEL
```python
import pickle
```
```pyhthon
filename = 'diabetes.sav'
pickle.dump(model, open(filename, 'wb'))
```

## Evaluation

```python
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
```
```python
print('akurasi data training :', training_data_accuracy)
```
0.7850162866449512
```python
X_test_prediction = model.predict(X_test)
test_data_accuracy =accuracy_score(X_test_prediction, Y_test)
```
```python
print('akurasi data testing :', test_data_accuracy)
```
0.7532467532467533

akurasi adalah metrik evaluasi yang mengukur seberapa baik model membuat prediksi yang benar dari total prediksi yang dilakukan. dalam konteks klasifikasi, akurasi memberikan gambaran mengenai seberapa sering model memprediksi kelas yang benar, baik kelas itu positif maupun negatif.
## Deployment
https://my-apk-estimasi-diabetes.streamlit.app//
