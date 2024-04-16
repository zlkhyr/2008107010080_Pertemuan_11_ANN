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
- **Data Kategorikal**
    
    <img width="288" alt="image" src="https://user-images.githubusercontent.com/116968275/216391713-709812c9-146b-4e2b-a8b9-482f57f9d7cd.png">
    
    Gambar 2. Visualisasi Penyebaran Data pada Kolom *mark*

- **Data Numerikal**
  
    <img width="666" alt="image" src="https://user-images.githubusercontent.com/116968275/216536393-e85da033-9caa-476f-9504-ee5f25bb10f8.png">
    
    Gambar 9. Visualisasi Penyebaran pada Data Numerikal

    Berikut beberapa informasi yang dapat diambil dari grafik di atas:
    - Dataset ini banyak mengandung data mobil dengan tahun produksi diatas tahun 2000.
    - Kebanyakan mobil pada *dataset* ini menempuh jarak berkisar dari 10 km hingga 224000 km.
    - Semakin rendah jarak tempuh mobil, maka harga mobil itupun semakin tinggi.
    - Dataset ini banyak mengandung mobil dengan volume mesin 1000cc hingga 2000cc.

### *Multivariate Analysis*:
Ini merupakan analisis yang digunakan untuk melihat korelasi antar kolom. Dalam analisis ini, data numerikal dan data kategorikal akan dipisahkan.
- **Data Kategorikal**
    - **Korelasi antara *price* dengan *mark***

      <img width="960" alt="image" src="https://user-images.githubusercontent.com/116968275/216543954-02269cc8-f9cf-43e9-b1a6-591521cd307e.png">
      
      Gambar 10. Visualisasi Korelasi antara Kolom *price* dengan Kolom *mark*

      Beberapa merek mobil memiliki pengaruh terhadap harga mobil, tetapi kebanyakan merek mobil tersebut merupakan sebuah *luxury brand* dimana harga mobilnya sudah pasti tinggi (*Mercedes Benz*, *BMW*, *Audi*, dan *Volvo*).
      
    - **Korelasi antara *price* dengan *fuel***

      <img width="960" alt="image" src="https://user-images.githubusercontent.com/116968275/216544047-b2b3f154-0176-4ba2-b709-6f2e5e7b89e9.png">
      
      Gambar 11. Visualisasi Korelasi antara Kolom *price* dengan Kolom *fuel*

      Jenis bahan bakar tidak memiliki pengaruh terhadap harga mobil.
      
    - **Korelasi antara *price* dengan *province***

      <img width="960" alt="image" src="https://user-images.githubusercontent.com/116968275/216544137-e0222b9f-f30a-491c-85e2-fab8c61cbee2.png">
      
      Gambar 12. Visualisasi Korelasi antara Kolom *price* dengan Kolom *province*

      Provinsi asal mobil tidak memiliki pengaruh terhadap harga mobil, walaupun ada beberapa provinsi yang memiliki harga mobil lebih tinggi dibandingkan dengan provinsi lain.

- **Data Numerikal**
  - ***Pairplot***

    <img width="449" alt="image" src="https://user-images.githubusercontent.com/116968275/216545855-072042ee-a49f-4f69-98f4-bfa726bb69c2.png">
    
    Gambar 13. Visualisasi Penyebaran Data dengan *Pairplot* pada Data Numerikal

    Berdasarkan *Pairplot* di atas, terlihat jika nilai kapasitas mesin(*vol_engine*) tidak memiliki pengaruh terhadap harga mobil, dikarenakan sebaran data yang terlihat acak dan tidak memiliki pola. Untuk tahun produksi mobil (*year*), terlihat jika semakin tinggi tahun mobil (Mendekati tahun sekarang) maka akan semakin tinggi pula harganya dan dibuktikan dengan adanya pola data positif. Untuk jarak tempuh mobil (*mileage*), datanya memiliki korelasi negatif dengan harga mobil yang dimana semakin tinggi jarak tempuh mobil maka harganya akan semakin rendah.
    
  - ***Heat Map***

    <img width="419" alt="image" src="https://user-images.githubusercontent.com/116968275/216545921-58f58eda-c559-49f4-817f-ec484be84ef7.png">
    
    Gambar 14. Visualisasi Korelasi Data Numerikal dengan *Heat Map* pada Data Numerikal

    Jika kita lihat berdasarkan *Heat Map* yang sudah dibuat, *Heat Map* ini menjadi bukti penguat dari apa yang sudah disimpulkan dari *Pairplot* yang telah dibuat. Terlihat jika nilai tahun produksi mobil (*year*) memiliki korelasi positif terhadap harga mobil dan jarak tempuh mobil (*mileage*) memiliki korelasi negatif terhadap harga mobil.
    
## *Data Preparation*
Dalam bagian ini, ada beberapa hal yang akan dilakukan guna menunjang data yang baik untuk digunakan untuk melatih model.
- ***One-Hot Encoding***

  *One-Hot Encoding* merupakan suatu teknik yang digunakan untuk mengubah data kategorikal menjadi data numerik, dikarenakan model hanya bisa mengenali satu jenis data saja. Teknik ini dilakukan dengan cara menggunakan fungsi `get_dummies()` yang dapat mengubah data kategorikal menjadi kolom baru dengan data 0 atau 1.
  
  Tabel 1. *Dataset Car Price in Poland* Setelah Menggunakan *One-Hot Encoding*
  
  |   | year | mileage | vol_engine | price | mark_alfa-romeo | mark_audi | mark_bmw | mark_chevrolet | mark_citroen | mark_fiat | ... | province_Mazowieckie | province_Małopolskie | province_Podkarpackie | province_Pomorskie | province_Warmińsko-mazurskie | province_Wielkopolskie | province_Zachodniopomorskie | province_Łódzkie | province_Śląskie | province_Świętokrzyskie |
  |--:|-----:|--------:|-----------:|------:|----------------:|----------:|---------:|---------------:|-------------:|----------:|----:|---------------------:|---------------------:|----------------------:|-------------------:|-----------------------------:|-----------------------:|----------------------------:|-----------------:|-----------------:|------------------------:|
  | 0 | 2015 |  139568 |       1248 | 35900 |               0 |         0 |        0 |              0 |            0 |         0 | ... |                    1 |                    0 |                     0 |                  0 |                            0 |                      0 |                           0 |                0 |                0 |                       0 |
  | 1 | 2018 |   31991 |       1499 | 78501 |               0 |         0 |        0 |              0 |            0 |         0 | ... |                    0 |                    0 |                     0 |                  0 |                            0 |                      0 |                           0 |                0 |                1 |                       0 |
  | 5 | 2017 |  121203 |       1598 | 51900 |               0 |         0 |        0 |              0 |            0 |         0 | ... |                    1 |                    0 |                     0 |                  0 |                            0 |                      0 |                           0 |                0 |                0 |                       0 |
  | 6 | 2017 |  119965 |       1248 | 44700 |               0 |         0 |        0 |              0 |            0 |         0 | ... |                    0 |                    0 |                     0 |                  0 |                            0 |                      0 |                           0 |                0 |                0 |                       0 |
    | 7 | 2016 |  201658 |       1248 | 29000 |               0 |         0 |        0 |              0 |            0 |         0 | ... |                    0 |                    0 |                     0 |                  0 |                            0 |                      0 |                           0 |                0 |                0 |                       0 |

- ***Train-Test Split***

  *Train-Test Split* merupakan suatu fungsi yang digunakan untuk memecahkan data menjadi data latih dan data uji. Dalam pengaplikasiannya, digunakan rasio antara data latih dan data uji guna menghindari model menjadi *overfit* atau *underfit*. Dalam kasus prediksi harga mobil ini, digunakan rasio 90:10 dikarenakan terdapat ~100000 data dalam *dataset* ini.
  
  Hasil dari pembagian data latih dan data uji dengan fungsi *Train-Test Split* dengan menggunakan rasio 90:10 adalah sebagai berikut:
  - Total Data Keseluruhan: 103712
  - Total Data Latih: 93340
  - Total Data Uji: 10372
    
- **Standarisasi Data**

  Model *machine learning* dilatih menggunakan data yang sudah diproses sebelumnya agar nantinya memiliki performa dan akurasi yang baik. Model ini dapat dilatih lebih baik dan cepat jika data yang sudah kita proses sebelumnya memiliki nilai yang seragam dan memiliki skala yang relatif sama. Untuk memenuhi hal tersebut dapat digunakan fungsi `Standard_Scaler()` dari *library sklearn*. Fungsinya adalah agar data yang akan kita pakai untuk model memiliki nilai *mean* = 0 dan nilai standar deviasi = 1.
  
  Tabel 2. Hasil Standarisasi Data menggunakan fungsi `Standard_Scaler()`
  
  |       | year       | mileage    | vol_engine |
  |-------|------------|------------|------------|
  | count | 93340.0000 | 93340.0000 | 93340.0000 |
  |  mean |    -0.0000 |     0.0000 |    -0.0000 |
  |  std  |     1.0000 |     1.0000 |     1.0000 |
  |  min  |    -8.8020 |    -1.5912 |    -1.3902 |
  |  25%  |    -0.6868 |    -0.7711 |    -0.5681 |
  |  50%  |     0.0346 |     0.0600 |    -0.0772 |
  |  75%  |     0.7559 |     0.6592 |     0.2506 |
  |  max  |     1.6576 |    29.4388 |     7.8233 |

## *Modeling*
Pada tahap ini, akan dilakukan pengujian dan pelatihan model menggunakan 3 algoritma (*K-Nearest Neighbor*, *Random Forest*, dan *AdaBoost*)

- ***K-Nearest Neighbor***

  *K-Nearest Neighbor* merupakan salah satu algoritma yang bekerja dengan cara membandingkan jarak satu sampel dengan sampel lain dengan cara memilih jumlah k tetangga terdekat. 
  
  Dalam algoritma *K-Nearest Neighbor*, terdapat *hyperparameter* yang dapat disesuaikan guna meningkatkan performa model:
  - `n_neighbor` : Merupakan jumlah k tetangga terdekat.

  Selain itu, terdapat beberapa kelebihan dan kekurangan dari algoritma *K-Nearest Neighbor* ini:
  - **Kelebihan**

    1. Metode ini cenderung lebih sederhana dan mudah dipahami dibandingkan dengan algoritma lain.
    2. Metode ini pun tidak memerlukan pemodelan, menjadikan algoritma ini dapat memangkas waktu dalam membangun model.

  - **Kekurangan**

    1. Kinerjanya yang terhitung lambat dikarenakan harus membandingkan jarak satu sampel dengan sampel yang lain.
    2. Prediksi dari metode ini dapat sangat berubah walaupun hanya merubah nilai `n_neighbor`, maka dari itu diperlukan nilai k yang pas dan sesuai dengan data yang digunakan.

- ***Random Forest***

  *Random Forest* merukapan algoritma yang bekerja dengan cara menjalankan beberapa model sekaligus dan membandingkan hasil akhirnya. Dalam penggunaannya, algoritma *Random Forest* menggunakan banyak *decision tree* yang dibangun secara acak dari dataset yang ada. Nantinya, hasil akhir dari algoritma ini adalah gabungan dari hasil akhir beberapa *decision tree* yang telah dijalankan. 
  
  Terdapat beberapa *hyperparameter* dalam algoritma ini:
  - `n_estimator` : Ini merupakan berapa banyak *decision tree* yang akan dijalankan dalam model.
  - `max_depth` : Merupakan nilai dari seberapa dalam *decision tree* akan melakukan percabangan.
  - `random_state` : Merupakan nilai untuk mengontrol *seed* dari model yang akan dijalankan.

  Ada juga kelebihan dan kekurangan dari algoritma ini:
  - **Kelebihan**

    1. Metode ini memiliki akurasi yang tinggi dikarenakan model menjalankan beberapa *decision tree* secara bersamaan, dan dapat menghindari masalah *overfit* pada model.

  - **Kekurangan**

    1. Dalam pelatihannya, algoritma *Random Forest* memerlukan waktu yang lebih lama dikarenakan menjalankan banyak *decision tree* secara bersamaan.
  
- ***AdaBoost***

  *AdaBoost (Adapive Boost)* merupakan salah satu algoritma yang memiliki cara kerja menggabungkan beberapa model yang nantinya dapat memperbaiki akurasi dan stabilitas dari model. *AdaBoost* bekerja dengan cara menempatkan bobot lebih kepada data yang sebelumnya tidak diprediksi benar oleh model sebelumnya, nantinya model selanjutnya akan memberi fokus lebih pada data tersebut dan akhirnya meningkatkan akurasi dari model tersebut. Proses ini dilakukan berulang hingga mencapai nilai akurasi yang tinggi.
  
  Ada beberapa *hyperparameter* yang dapat diatur guna meningkatkan performa dari algoritma ini:
  - `learning_rate` : Merupakan nilai dari laju pembelajaran model.
  - `random_state` : Merupakan nilai untuk mengontrol *seed* dari model yang akan dijalankan.

  Adapun kelebihan dan kekurangan dari algoritma ini:
  - **Kelebihan**

    1. Metode ini memiliki akurasi yang tinggi jika dibandingkan dengan metode sederhana lainnya (Contohnya seperti *Random Forest*).

  - **Kekurangan**

    1. Algoritma ini biasanya rentan dengan data *outlier* dan membutuhkan waktu lama dalam proses latih dikarenakan menggabungkan beberapa jenis model dalam penggunaannya.

Dengan menggunakan ketiga algortima diatas serta mencari nilai *hyperparameter* yang paling baik, didapatkan hasil sebagai berikut:

Tabel 3. Hasil Nilai *Hyperparameter* Terbaik pada Setiap Algoritma

| Model               | *Hyperparameter* Terbaik                                   |
|---------------------|------------------------------------------------------------|
| K-Nearest Neighbor  | `n_neighbor` : 4                                           |
| AdaBoost            | `learning_rate` : 0.08, `random_state` : 55                |
| Random Forest       | `n_estimator` : 50, `max_depth` : 64, `random_state` : 55  |

## Evaluation
Untuk mengevaluasi hasil kinerja model yang telah dibangun, dalam kasus ini digunakan metrik MSE (*Mean Squared Error*). Metrik ini bekerja dengan cara melihat selisih dari nilai asli (*y_true*) dengan hasil prediksi dari masing-masing model. 

![image](https://user-images.githubusercontent.com/116968275/216582083-3a912459-5489-4e29-a7c8-4f7d0b3a12c0.png)

Gambar 15. Rumus Mencari Nilai *Mean Squared Error*

**Penjelasan**:
- MSE : *Mean Squared Error*
- yi : Nilai asli (*y_true*)
- y_pred : Nilai prediksi model

Dengan menggunakan MSE, berikut hasil dari setiap model algortima yang digunakan:

Tabel 4. Hasil Nilai *Error* pada Data Uji dan Data Latih Setiap Model Algoritma

| Model               | Train            | Test             |
|---------------------|------------------|------------------|
| K-Nearest Neighbor  | 520014.135924    | 732296.875684    |
| AdaBoost            | 1838754.256309   | 1703197.195434   |
| Random Forest       | 184651.209957    | 589524.177038    |

<img width="339" alt="image" src="https://user-images.githubusercontent.com/116968275/216583247-26c4840a-e6c8-4a5f-b40b-def19d131a00.png">

Gambar 16. Visualisasi *error* pada Data Latih dan Data Uji menggunakan Model Algoritma yang Sudah Dilatih

Berikut juga hasil prediksi dari setiap model jika dibandingkan dengan nilai *y_true* dari *dataset*:

Tabel 5. Hasil Nilai Prediksi Harga Mobil pada Setiap Model Algoritma

| Model               | Hasil Prediksi   |
|---------------------|------------------|
| y_true              | 76000            |
| K-Nearest Neighbor  | 51625            |
| AdaBoost            | 78337.3          |
| Random Forest       | 52459.8          |

Melihat hasil di atas, maka model yang menggunakan algoritma *AdaBoost* merupakan model dengan nilai akurasi yang paling tinggi serta memiliki nilai error yang paling kecil jika dibandingkan dengan model *K-Nearest Neighbor* dan *Random Forest*.
