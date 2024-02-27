# Laporan Proyek 1 Machine Learning Mahir - Andika Rahman Teja

## Domain Proyek

*Wine* merupakan minuman berbasis alkohol yang pembuatannya melalui proses fermentasi [1]. *Wine* juga menjadi minuman yang sangat digemari oleh masyarakat di benua Eropa (Italia, Prancis, dan Spanyol) dan Amerika (Amerika Serikat, Chile, dan Argentina). Hal ini dapat dibuktikan dari kegiatan ekspor dan impor *wine* di negara-negara tersebut yang selalu meningkat dari tahun 1995 [2 - 3]. Tidak hanya di benua Eropa dan Amerika, hampir semua masyarakat di dunia juga menghasilkan dan mengonsumsi minuman *wine*. Terbukti pada tahun 2021, produksi *wine* secara global mencapai angka 250 juta hektoliter. Dengan angka tersebut dapat membuktikan bahwa produksi *wine* dapat menjadi potensi bisnis yang besar. Selain bisnis di sektor industri makanan, *wine* juga berpontensi untuk digunakan pada bisnis pertanian, farmasi, hingga kosmetik [4].

Namun, untuk dapat memproduksi *wine* dengan kualitas yang baik sehingga dapat dipasarkan secara global tidaklah mudah. Ada banyak faktor yang mempengaruhi sebuah *wine* sehingga dapat dinikmati dengan baik oleh pelanggan. Faktor-faktor tersebut mulai dari tingkat keasaman, persentase kandungan gula, densitas (kepadatan), hingga faktor banyaknya kandungan alkohol pada minuman tersebut [5]. Dengan banyaknya faktor tersebut, akan sulit bagi produsen, penjual hingga pembeli untuk membedakan kualitas *wine* yang baik dan buruk. Untuk menyelesaikan masalah tersebut, penulis akan membuat sebuah pendekatan ilmiah melalui salah satu penerapan *Artificial Intelligence* (AI), yakni dengan membuat beberapa model *machine learning* dan memilih model terbaik yang mampu melakukan klasifikasi kualitas *wine* berdasarkan faktor-faktor yang telah dijelaskan sebelumnya. 


## Business Understanding

### Rumusan Masalah
- Bagaimana cara mengetahui kualitas *wine* dari kandungan yang ada pada *wine* tersebut?
- Apa model *machine learning* terbaik yang dapat mengklasifikasikan kualitas *wine*?

### Tujuan
- Untuk mengetahui kualitas *wine* dari kandungan yang ada pada *wine* tersebut
- Untuk mengetahui model *machine learning* terbaik yang dapat mengklasifikasikan kualitas *wine*

### Dampak Penelitian terhadap Bisnis
- Membantu proses produksi *wine*, terutama pada proses klasifikasi kualitas *wine*
- Membantu produsen untuk menentukan harga *wine* yang diproduksi berdasarkan kualitasnya
- Meningkatkan kepuasan pelanggan terhadap produk *wine* karena telah membeli sesuai kualitas yang ditawarkan

### Solusi yang ditawarkan
- Menggunakan model *Adaptive Boosting* berupa `AdaBoostClassifier` beserta *hyperparameter tuning*-nya
- Menggunakan model *Gradient Boosting* berupa `XGBClassifier` beserta *hyperparameter tuning*-nya
- Menggunakan model *Artificial Neural Network* (ANN) berupa struktur `sequential`


## Data Understanding

Adapun dataset yang digunakan untuk penelitian ini adalah dataset yang bersifat *open source* di [Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009). Selain itu, dataset tersebut juga tersedia di [UCI Machine Learning](https://archive.ics.uci.edu/dataset/186/wine+quality). Dataset tersebut semula memiliki informasi sebagai berikut:

|Describe|fixed acidity|volatile acidity|citric acid|residual sugar|chlorides|free sulfur dioxide|total sulfur dioxide|density|pH|sulphates|alcohol|quality|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|count|1599|1599|1599|1599|1599|1599|1599|1599|1599|1599|1599|1599|
|mean|8.32|0.53|0.27|2.54|0.09|15.87|46.47|1.00|3.31|0.66|10.42|5.64|
|std|1.74|0.18|0.19|1.41|0.05|10.46|32.90|0.00|0.15|0.17|1.07|0.81|
|min|4.60|0.12|0.00|0.90|0.01|1.00|6.00|0.99|2.74|0.33|8.40|3.00|
|25%|7.10|0.39|0.09|1.90|0.07|7.00|22.00|1.00|3.21|0.55|9.50|5.00|
|50%|7.90|0.52|0.26|2.20|0.08|14.00|38.00|1.00|3.31|0.62|10.20|6.00|
|75%|9.20|0.64|0.42|2.60|0.09|21.00|62.00|1.00|3.40|0.73|11.10|6.00|
|max|15.90|1.58|1.00|15.50|0.61|72.00|289.00|1.00|4.01|2.00|14.90|8.00|

Tabel 1. Statistik Data Pada Dataset **Red Wine Quality**

Adapun keterangan *describe* pada Tabel 1 adalah sebagai berikut:
- `count` = Banyaknya data
- `mean` = rata-rata data
- `std` = standar deviasi data
- `min` = nilai minimum pada data
- `25%` = kuartil 1 pada data
- `50%` = kuartil 2 / nilai tengah pada data
- `75%` = kuartil 3 pada data
- `max` = nilai maksimal pada data

Selain itu, terdapat informasi berupa korelasi antar fitur pada dataset ini. Berikut visualisasi korelasi antar fitur.

![heatmap_corr](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%201/images/Heatmap_corr.png?raw=true)

Gambar 1. *Heatmap Correlation* pada Dataset **Red Wine Quality**

Dari informasi-informasi di atas, dapat disimpulkan bahwa:
- Tidak ada data yang **NULL** (total data = 1599)
- Terdapat 12 fitur / kolom dan sesuai keterangan pemilik dataset bahwa fitur `quality` akan menjadi targetnya
- Semua data berbentuk numerik (*float* dan *int*)
- Fitur `quality` ternyata hanya bernilai dari 3 hingga 8 (tidak sesuai keterangan pemilik dataset bahwa bernilai 0 hingga 10)
- Fitur `pH` bernilai 2.74 hingga 4.01 sehingga menunjukkan bahwa dataset ini sesuai kondisi riil (*wine* bersifat asam)
- Fitur `residual sugar` dan `free sulfur dioxide` memiliki korelasi yang rendah terhadap fitur target, yakni `quality`

### Fitur-Fitur Pada Dataset:
- `fixed acidity` =  jumlah asam yang tetap dalam *wine* setelah semua asam yang mudah menguap telah dihilangkan
- `volatile acidity` = jumlah asam yang mudah menguap dalam *wine*
- `citric acid` = komponen asam yang ditemukan dalam buah-buahan, termasuk anggur
- `residual sugar` = jumlah gula yang tersisa setelah proses fermentasi selesai
- `chlorides` = kandungan garam klorida dalam *wine*
- `free sulfur dioxide` = sulfur dioksida dalam bentuk bebas yang terlarut dalam *wine*
- `total sulfur dioxide` = jumlah total sulfur dioksida dalam *wine*, termasuk yang terikat dengan senyawa lain
- `density` = massa *wine* per satuan volume pada suhu dan tekanan tertentu (kepadatan *wine*)
- `pH` = ukuran keasaman atau kebasaan pada *wine*
- `sulphates` = kandungan sulfur pada *wine*
- `alcohol` = kandungan alkohol pada *wine*
- `quality` [**Output Variabel**] = kualitas *wine*


## Data Preparation
1. Menjalankan fungsi `df.info()` dan `df.describe()` untuk mengatahui jumlah data dan fitur / kolom data, memastikan apakah data memiliki nilai NULL, dan mengetahui tipe data tiap fitur. 
2. Melakukan visulisasi data untuk mengetahui persebaran dan kondisi dari data. Berikut ini hasil visualisasi data berupa histogram dan *boxplot*.

![histogram_before](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%201/images/Histogram(before).png?raw=true)

Gambar 2. Histogram pada Dataset **Red Wine Quality**

![boxplot_before](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%201/images/Boxplot(before).png?raw=true)

Gambar 3. *Boxplot* pada Dataset **Red Wine Quality**

<<<<<<< HEAD
3. Mengatasi data yang *skewed* atau memiliki banyak *outlier* dengan menggunakan metode IQR (*interquartile range*). Berikut ini rumus dari metode IQR.
- $$\text{Batas Bawah} = Q1 - (1.5 \times \text{IQR})$$
- $$\text{Batas Atas} = Q3 + (1.5 \times \text{IQR})$$
=======
3. Mengatasi data yang *skewed* atau memiliki banyak *outlier* dengan menggunakan metode IQR (*interquartile range*). Berikut ini rumus dari metode IQR dan implementasinya pada bahasa pemrograman *python*.
- $\text{Batas Bawah} = Q1 - (1.5 \times \text{IQR})$
- $\text{Batas Atas} = Q3 + (1.5 \times \text{IQR})$
>>>>>>> d35070a1cb5b65219f471d60d343f09fe4c6b63e

Hal ini bertujuan untuk membersihkan data agar model *machine learning* dapat memahami data yang ada dengan lebih baik. Berikut ini hasil penerapan metode IQR pada dataset **Red Wine Quality**.

|Fitur / Kolom|Total Data (Tidak NULL)|Tipe Data|
|---|---|---|
|*fixed acidity*|1235|*float64*|
|*volatile acidity*|1235|*float64*|
|*citric acid*|1235|*float64*|
|*residual sugar*|1235|*float64*|
|*chlorides*|1235|*float64*|
|*free sulfur dioxide*|1235|*float64*|
|*total sulfur dioxide*|1235|*float64*|
|*density*|1235|*float64*|
|pH|1235|*float64*|
|*sulphates*|1235|*float64*|
|*alcohol*|1235|*float64*|
|*quality*|1235|*int64*|

Tabel 2. *overview* Data pada Dataset **Red Wine Quality** Setelah Menerapkan Metode IQR

![histogram_after](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%201/images/Histogram(after).png?raw=true)

Gambar 4. Histogram pada Dataset **Red Wine Quality** Setelah Menerapkan Metode IQR

![boxplot_after](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%201/images/Boxplot(after).png?raw=true)

Gambar 5. *Boxplot* pada Dataset **Red Wine Quality** Setelah Menerapkan Metode IQR

4. Melakukan seleksi fitur dengan mempertimbangkan korelasi dari fitur `quality` dengan fitur lainnya. Korelasi fitur yang dilakukan menggunakan metode *Pearson Correlation* dengan menjalankan fungsi `df.corr()` yang juga divisualisasikan menggunakan *library* **seaborn**. Tujuan seleksi fitur ini untuk mengurangi dimensi data (agar tidak terjadi *overfitting*) dan meningkatkan akurasi model agar dapat memahami data yang relevan terhadap fitur target (yakni `quality`)
5. Melakukan proses *labeling* pada fitur `quality` untuk mempermudah klasifikasi. Berikut perbandingan data sebelum dan setelah dilakukan proses *labeling*.

![jumlah_wine](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%201/images/Jumlah_wine.png?raw=true)

Gambar 6. Data pada Fitur `quality` Sebelum Proses *Labeling*

![jumlah_wine_kategorikal](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%201/images/Jumlah_wine(kategorikal).png?raw=true)

Gambar 7. Data pada Fitur `quality` Setelah Proses *Labeling*

6. Memisah dataset menjadi 2, yakni **X** sebagai variabel *input* berupa semua fitur dari dataset kecuali fitur `quality` dan **y** sebagai variabel *output* berupa fitur `quality`. Variabel **X** akan dilakukan standarisasi menggunakan fungsi `StandardScaler`. Hal ini bertujuan untuk memudahkan model dalam melakukan prediksi
7. Variabel **X** dan **y** masing-masing akan dipisah menjadi 2 bagian, yakni bagian **train** dan **valid** dengan perbandingkan data train:valid = 7:3. Adapun pembagian data tersebut menggunakan *library* **sklearn**. Tujuan pembagian data tersebut untuk menguji akurasi model yang ada.


## Modeling

### Model AdaBoostClassifier
**Kelebihan**
- Mengurangi kemungkinan terjadi *overfitting* karena model fokus pada pengurangan bias dan varians
- Parameter yang tidak rumit

**Kekurangan**
- Sensitif terhadap data *noise* atau *outlier*

**Parameter yang disesuaikan**
- `learning_rate` = mengatur bobot yang diterapkan pada setiap regressor di masing-masing proses iterasi boosting
- `n_estimators` = jumlah model yang dilakukan proses *ensemble*

|Parameter|Rentang Parameter|
|---|---|
|n_estimators|50 - 200|
|learning_rate|0.01, 0.05, 0.1, 0.3, 1|

Tabel 3. Parameter dan Rentang Parameter yang Disesuaikan pada Model **AdaBoostClassifier**

Adapun proses *hyperparameter tuning* ini menggunakan metode **RandomizedSearchCV**

### Model XGBClassifier
**Kelebihan**
- Mampu menangani data yang besar dan kompleks
- Mendukung proses *training* model secara paralel

**Kekurangan**
- Rentan mengalami *overfitting*
- Memiliki parameter yang kompleks dan rumit

**Parameter yang disesuaikan**
- `learning_rate` = mengatur bobot yang diterapkan pada setiap regressor di masing-masing proses iterasi boosting
- `max_depth` = Mengatur maksimum kedalaman tiap *decision tree*
- `min_child_weight` = mengatur *weight* atau bobot minimum yang dibutuhkan untuk membuat *child* pada *decision tree*
- `subsample` = Proporsi dari sampel *training* yang akan digunakan untuk melatih setiap *decision tree*
- `colsample_bytree` = Proporsi dari fitur-fitur yang akan digunakan dalam membuat setiap *decision tree*
- `n_estimators` = jumlah *decision tree* yang dilakukan pada proses ensemble

|Parameter|Rentang Parameter|
|---|---|
|learning_rate|0.01, 0.1, 0.2, 0.3, 0.5|
|max_depth|3, 5, 6, 10|
|min_child_weight|1, 3, 5|
|subsample|0.5, 0.7, 1|
|colsample_bytree|0.5, 0.7, 1|
|n_estimators|100, 200, 500|
Tabel 4. Parameter dan Rentang Parameter yang Disesuaikan pada Model **XGBClassifier**

Adapun proses *hyperparameter tuning* ini menggunakan metode **RandomizedSearchCV**

### Model ANN
**Kelebihan**
- Implementasi yang mudah
- Fleksibel untuk mengatur parameter

**Kekurangan**
- Rentan mengalami *overfitting*
- Biaya komputasi cenderung mahal (membutuhkan *gpu* pada beberapa kasus)

**Arsitektur Model**

|Urutan *Layer*|Tipe *Layer*|Ukuran *Output*|Jumlah Parameter|
|---|---|---|---|
|1|Dense|(None, 128)|1280|
|2|Dropout|(None, 128)|0|
|3|Dense|(None, 512)|66048|
|4|Dropout|(None, 512)|0|
|5|Dense|(None, 512)|262656|
|6|Dropout|(None, 512)|0|
|7|Dense|(None, 512)|262656|
|8|Dropout|(None, 512)|0|
|9|Dense|(None, 1)|513|

Tabel 5. Arsitektur ANN Struktur *Sequential*


## Evaluation

### Hasil *Hyperparameter Tuning*

Berikut ini hasil *hyperparameter tuning* untuk model **AdaBoostClassifier* dan **XGBClassifier**.

|Parameter|Ukuran Parameter|
|---|---|
|learning_rate|0.1|
|n_estimators|71|

Tabel 6. Parameter **AdaBoostClassifier** Hasil *Hyperparameter Tuning*

|Parameter|Ukuran Parameter|
|---|---|
|learning_rate|0.01|
|max_depth|5|
|min_child_weight|1|
|subsample|0.7|
|colsample_bytree|1|
|n_estimators|500|

Tabel 7. Parameter **XGBClassifier**

### Model Terbaik

Karena model ANN dilatih pada setiap *epoch* sehingga akurasinya dapat dicatat melalui sebuah grafik plot akurasi dan *loss* sebagai berikut.

![Akurasi_ANN](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%201/images/Model_ANN_accuracy.png?raw=true)

Gambar 8. Plot Akurasi Model ANN

![Loss_ANN](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%201/images/Model_ANN_loss.png?raw=true)

Gambar 9. Plot *Loss* Model ANN

Setelah dilakukan beberapa kali percobaan, didapatkan akurasi prediksi dari ketiga model yang telah diuji sebagai berikut.
|Model|Akurasi|
| --- | --- |
|AdaBoostClassifier sebelum *hyperparameter tuning*|0.74|
|AdaBoostClassifier setelah *hyperparameter tuning*|0.74|
|XGBClassifier sebelum *hyperparameter tuning*|0.75|
|XGClassifier setelah *hyperparameter tuning*|**0.76**|
|ANN struktur *sequential*|0.75|

Tabel 8. Akurasi Seluruh Model yang Diujikan

Dari tabel tersebut, dapat disimpulkan bahwa model **XGBClassifier** dalam kondisi telah dilakukan *hyperparameter tuning* merupakan model terbaik untuk melakukan klasifikasi kualitas *wine* karena memiliki akurasi sebesar $\pm$ 76%.


Metrik evaluasi yang digunakan pada penelitian ini adalah menggunakan fungsi dari *library* **Scikit-learn** (**sklearn**), yakni `classification_report`. Alasan menggunakan fungsi tersebut adalah untuk mempermudah mendapatkan informasi terkait metrik seperti **precision**, **recall**, **F1 score**, dan **akurasi**. Berikut ini penjelasan mengenai keempat metrik tersebut:
- **Precision** = mengukur seberapa banyak dari item yang diprediksi sebagai positif benar-benar benar positif. Berikut ini adalah rumus dari *precision*.
$$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$

- **Recall** = mengukur seberapa banyak dari semua item yang benar-benar positif yang berhasil diidentifikasi oleh model. Berikut ini adalah rumus dari *recall*.
$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$

- **F1 score** =  rata-rata harmonik dari precision dan recall. Berikut ini rumus dari *F1 score* $$\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **Akurasi** = rasio dari jumlah prediksi yang benar (*True Positives* dan *True Negatives*) dibagi dengan jumlah total prediksi. Berikut ini rumus dari akurasi.
$$\text{Akurasi} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Samples}}$$

Selain menggunakan `classification_report`, penulis juga menggunakan metrik tambahan berupa `confusion_matrix` untuk mengetahui detail hasil prediksi yang dilakukan oleh model. Berikut visualisasi dari `confusion_matrix`.

![confusion_matrix](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%201/images/Confusion_matrix.png?raw=true)

Gambar 10. *Confusion Matrix* Hasil Prediksi Model **XGBClassifier** Setelah Dilakukan *Hyperparameter Tuning*

Pada Gambar 10, memiliki penjelasan sebagai berikut.
- Model memprediksi **benar** kualitas *wine* 0 atau buruk sebanyak 118 kali
- Model memprediksi **benar** kualitas *wine* 1 atau bagus sebanyak 165 kali
- Model memprediksi **salah** kualitas *wine* yang seharusnya 0 namun diprediksi sebagai 1 sebanyak 36 kali
- Model memprediksi **salah** kualitas *wine* yang seharusnya 1 namun diprediksi sebagai 0 sebanyak 52 kali

Dengan akurasi model tersebut, diharapkan dapat membantu produsen, penjual, dan pembeli *wine* untuk membantu membedakan kualitas *wine* yang bagus dan yang buruk.


# Referensi
1. F. Y. Wiredjo, "WINE PAIRING TERHADAP DELAPAN MAKANAN NUSANTARA INDONESIA (RENDANG, SOTO BETAWI, GUDEG, RAWON, BAKSO, SATE AYAM, PEMPEK, AYAM BETUTU)", Universitas Katholik Soegijapranata, Semarang, 2021.
2. N. Obermayer, E. Kővári, J. Leinonen, G. Bak, and M. Valeri, “How social media practices shape family business performance: The Wine Industry Case Study,” European Management Journal, vol. 40, no. 3, pp. 360–371, Jun. 2022. 
3. A. A. Ugaglia, J.-M. Cardebat, and A. Corsi, The Palgrave Handbook of Wine Industry Economics. Cham, Switzerland: Palgrave Macmillan, 2019. 
4. R. Ferrer-Gallego and P. Silva, “The wine industry by-products: Applications for Food Industry and Health Benefits,” Antioxidants, vol. 11, no. 10, p. 2025, Oct. 2022.
5. K. MacNeil, The Wine Bible. New York: Workman Publishing Company, 2022.

**---Sekian, Terima Kasih---**