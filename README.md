# Predict whether someone is at high risk of a heart attack or not - *Artificial Neural Networks* (ANN) vs *Support Vector Machine* (SVM)
**Oleh Zul Akhyar**

Berikut merupakan Proyek membangun model untuk memprediksi apakah seseorang memiliki kemungkinan serangan jantung yang tinggi atau tidak

## Domain Proyek
### Latar Belakang
<br>
<div><img src="https://www.lenmed.co.za/wp-content/uploads/How-do-you-know-if-you-are-having-a-heart-attack.png"></div>
<br>

Deteksi dini untuk serangan jantung sangatlah penting. seperti, perawatan yang lebih baik, dan penurunan biaya kesehatan. Dengan mengidentifikasi individu berisiko tinggi secara dini, langkah-langkah pencegahan dapat diterapkan untuk mengurangi kemungkinan serangan jantung. Ini tidak hanya meningkatkan kesehatan individu, tetapi juga memiliki dampak positif pada kesehatan masyarakat secara keseluruhan.

Disini saya akan membuat dua model untuk memprediksi risiko serangan jantung: satu menggunakan **Artificial Neural Networks (ANN)** dan yang lainnya menggunakan **Support Vector Machine (SVM)**. Setelah kedua model selesai, saya akan membandingkan akurasi keduanya untuk menentukan model mana yang lebih baik dalam memprediksi risiko serangan jantung.

## *Data Understanding*
Data yang digunakan disini adalah data yang berisi informasi terkait serangan jantung yang didapat dari kaggle

*Link* data: [Disini](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)

### Berikut merupakan beberapa kolom yang terdapat dalam dataset:
1. **Age**: Age of the patient.
2. **Sex**: Gender of the patient.
3. **exang**: Exercise induced angina (1 = yes; 0 = no).
4. **ca**: Number of major vessels (0-3).
5. **cp**: Chest pain type:
   - Value 1: Typical angina
   - Value 2: Atypical angina
   - Value 3: Non-anginal pain
   - Value 4: Asymptomatic
6. **trtbps**: Resting blood pressure (in mm Hg).
7. **chol**: Cholesterol level in mg/dl fetched via BMI sensor.
8. **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).
9. **rest_ecg**: Resting electrocardiographic results:
   - Value 0: Normal
   - Value 1: Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
   - Value 2: Showing probable or definite left ventricular hypertrophy by Estes' criteria
10. **thalach**: Maximum heart rate achieved.
11. **target**: Indicates the likelihood of a heart attack:
    - 0 = Less chance of heart attack
    - 1 = More chance of heart attack

### *EDA (Exploratory Data Analysis)*:
- **Data Kategorikal (Count Plot)**
    
    <img alt="image" src="https://github.com/zlkhyr/2008107010080_Pertemuan_11_ANN/blob/main/img/plot1.png">

     - Melihat jumlah data untuk tiap kategori dalam kolom/fitur

- **Data Numerikal (Histogram)**
  
    <img alt="image" src="https://github.com/zlkhyr/2008107010080_Pertemuan_11_ANN/blob/main/img/plot2.png">
    
     - Melihat distribusi frekuensi dari data numerik
       
- **Korelasi (Heatmap)**
  
    <img alt="image" src="https://github.com/zlkhyr/2008107010080_Pertemuan_11_ANN/blob/main/img/plot3.png">
    
    - Melihat korelasi antar kolom/fitur terutama dengan fitur target/class

### *Outliers*:
   <img alt="image" src="https://github.com/zlkhyr/2008107010080_Pertemuan_11_ANN/blob/main/img/plot4.png">
    
   - Outliers pada data tidak terlalu signifikan
     
## *Data Preparation*
Dalam bagian ini, ada beberapa hal yang akan dilakukan guna menunjang data yang baik untuk digunakan untuk melatih model.
- ***Train-Test Split***

  *Train-Test Split* merupakan suatu fungsi yang digunakan untuk memecahkan data menjadi data latih dan data uji. Dalam pengaplikasiannya, digunakan rasio antara data latih dan data uji guna menghindari model menjadi *overfit* atau *underfit*. Dalam kasus ini, digunakan rasio 80:20 dimana 80% data latih dan 20% sisanya untuk data uji 
    
- **Feature scaling**

  Feature scaling adalah proses normalisasi atau standarisasi nilai-nilai fitur (fitur atau variabel independen) dalam dataset. Tujuan utama dari feature scaling adalah untuk menjaga rentang nilai setiap fitur agar seimbang, sehingga algoritma pembelajaran mesin dapat bekerja lebih baik dan konvergen lebih cepat.

## *Modeling*
Pada tahap ini, akan dilakukan pengujian dan pelatihan model menggunakan 2 algoritma (**Artificial Neural Networks (ANN)** dan **Support Vector Machine (SVM)**)

Dengan menggunakan kedua algortima diatas serta mencari nilai *hyperparameter* yang paling baik, didapatkan hasil sebagai berikut:

Tabel 3. Hasil Nilai *Hyperparameter* Terbaik pada Setiap Algoritma

| Model               | *Hyperparameter* Terbaik                                   |
|---------------------|------------------------------------------------------------|
| SVM                 | `C = 1`, `gamma = 0.1`, `random_state = 42`                |
|                     |                                                            |
| ANN                 | `unit = 6`, `kernel_initializer = 'uniform'`,              |
|                     | `activation = 'relu'('sigmoid' untuk output layer)`,       |
|                     | `optimizer = 'adam'`, `loss = 'binary_crossentropy'`,      |
|                     | untuk model fit : `batch_size = 10`, `epochs = 100`        |

## Evaluation
Untuk mengevaluasi hasil kinerja model yang telah dibangun, karena dalam kasus ini model yang dibangun adalah model klasifikasi maka akan digunakan confusion matrix dan clasification report dari sklearn. 
### Support Vector Machine (SVM)
<img alt="image" src="https://github.com/zlkhyr/2008107010080_Pertemuan_11_ANN/blob/main/img/plot5.png">

Model SVM cukup baik dalam memprediksi data TF (true positif), TN (true negatif) `[26, 29]` dan menghindari FN (false negatif), FP (false positif) `[3,3]`

|           | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
|     0     |   0.90    |  0.90  |   0.90   |    29   |
|     1     |   0.91    |  0.91  |   0.91   |    32   |
|           |           |        |          |         |
| **Accuracy** |           |        |   0.90   |    61   |
| **Macro Avg**|   0.90    |  0.90  |   0.90   |    61   |
|**Weighted Avg**|  0.90    |  0.90  |   0.90   |    61   |

`Model SVM memiliki akurasi 90%`

### Artificial Neural Networks (ANN)
<img alt="image" src="https://github.com/zlkhyr/2008107010080_Pertemuan_11_ANN/blob/main/img/plot6.png">

Model ANN juga hampis sama baik dalam memprediksi data TF (true positif), TN (true negatif) `[25, 29]` dan menghindari FN (false negatif), FP (false positif) `[3,4]` hannya bebeda pada sedikit pada niali false positif

|           | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
|     0     |   0.89    |  0.86  |   0.88   |    29   |
|     1     |   0.88    |  0.91  |   0.89   |    32   |
|           |           |        |          |         |
| **Accuracy** |           |        |   0.89   |    61   |
| **Macro Avg**|   0.89    |  0.88  |   0.88   |    61   |
|**Weighted Avg**|  0.89    |  0.89  |   0.89   |    61   |

`Model ANN memiliki akurasi 89%`

**Penjelasan**:

<img width="339" alt="image" src="https://assets-global.website-files.com/6266b596eef18c1931f938f9/644af6a24701d43aaecd8771_classification_guide_apc09.png">

- TP adalah jumlah sampel positif yang diprediksi dengan benar oleh model.
- TN adalah jumlah sampel negatif yang diprediksi dengan benar oleh model.
- FP adalah jumlah sampel negatif yang salah diprediksi sebagai positif oleh
- FN adalah jumlah sampel positif yang salah diprediksi sebagai negatif oleh 

Berdasarkan hasil di atas, model yang menggunakan algoritma **Support Vector Machine (SVM)** memiliki nilai akurasi yang lebih tinggi dibandingkan dengan model yang menggunakan **Artificial Neural Networks (ANN)**. Hal ini mungkin disebabkan oleh beberapa faktor, salah satunya adalah `ukuran dataset yang kecil`.

Ketika menggunakan dataset kecil, `SVM` mungkin `lebih efektif` karena cenderung `tidak memerlukan sebanyak data untuk melatih` modelnya. `SVM` juga cenderung `lebih baik dalam menangani dataset yang memiliki dimensi tinggi (banyak fitur)`, meskipun dataset tersebut kecil. Di sisi lain, `ANN` mungkin `memerlukan lebih banyak data untuk melatih dengan efektif`, terutama jika arsitektur jaringan yang kompleks digunakan. 
