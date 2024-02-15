# Laporan Proyek 1 Machine Learning - Andika Rahman Teja

## Domain Proyek

*Wine* merupakan minuman berbasis alkohol yang pembuatannya melalui proses fermentasi [1]. *Wine* juga menjadi minuman yang sangat digemari oleh masyarakat di benua Eropa (Italia, Prancis, dan Spanyol) dan Amerika (Amerika Serikat, Chile, dan Argentina). Hal ini dapat dibuktikan dari kegiatan ekspor dan impor *wine* di negara-negara tersebut yang selalu meningkat dari tahun 1995 [2 - 3]. Tidak hanya di benua Eropa dan Amerika, hampir semua masyarakat di dunia juga menghasilkan dan mengonsumsi minuman *wine*. Terbukti pada tahun 2021, produksi *wine* secara global mencapai angka 250 juta hektoliter. Dengan angka tersebut dapat membuktikan bahwa produksi *wine* dapat menjadi potensi bisnis yang besar. Selain bisnis di sektor industri makanan, *wine* juga berpontensi untuk digunakan pada bisnis pertanian, farmasi, hingga kosmetik [4].

Namun, untuk dapat memproduksi *wine* dengan kualitas yang baik sehingga dapat dipasarkan secara global tidaklah mudah. Ada banyak faktor yang mempengaruhi sebuah *wine* sehingga dapat dinikmati dengan baik oleh pelanggan. Faktor-faktor tersebut mulai dari tingkat keasaman, persentase kandungan gula, densitas (kepadatan), hingga faktor banyaknya kandungan alkohol pada minuman tersebut [5]. Dengan banyaknya faktor tersebut, akan sulit bagi produsen, penjual hingga pembeli untuk membedakan kualitas *wine* yang baik dan buruk. Untuk menyelesaikan masalah tersebut, penulis akan membuat sebuah pendekatan ilmiah melalui salah satu penerapan *Artificial Intelligence* (AI), yakni dengan membuat beberapa model *machine learning* dan memilih model terbaik yang mampu melakukan klasifikasi kualitas *wine* berdasarkan faktor-faktor yang telah dijelaskan sebelumnya. 

### Referensi
1. F. Y. Wiredjo, "WINE PAIRING TERHADAP DELAPAN MAKANAN NUSANTARA INDONESIA (RENDANG, SOTO BETAWI, GUDEG, RAWON, BAKSO, SATE AYAM, PEMPEK, AYAM BETUTU)", Universitas Katholik Soegijapranata, Semarang, 2021.
2. N. Obermayer, E. Kővári, J. Leinonen, G. Bak, and M. Valeri, “How social media practices shape family business performance: The Wine Industry Case Study,” European Management Journal, vol. 40, no. 3, pp. 360–371, Jun. 2022. 
3. A. A. Ugaglia, J.-M. Cardebat, and A. Corsi, The Palgrave Handbook of Wine Industry Economics. Cham, Switzerland: Palgrave Macmillan, 2019. 
4. R. Ferrer-Gallego and P. Silva, “The wine industry by-products: Applications for Food Industry and Health Benefits,” Antioxidants, vol. 11, no. 10, p. 2025, Oct. 2022.
5. K. MacNeil, The Wine Bible. New York: Workman Publishing Company, 2022. 


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
- Menggunakan model *Artificial Neural Network* berupa model `sequential`


## Data Understanding

Adapun dataset yang digunakan untuk penelitian ini adalah dataset yang bersifat *open source* di [Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009). Selain itu, dataset tersebut juga tersedia di [UCI Machine Learning](https://archive.ics.uci.edu/dataset/186/wine+quality). Dataset tersebut semula memiliki informasi sebagai berikut:
![df](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%201/images/df.png?raw=true)
Gambar 1. *Overview* Dataset **Red Wine Quality**

![df_info](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%201/images/df_info(before).png?raw=true)
Gambar 2. Jumlah Data dan Tipe Data pada Dataset **Red Wine Quality**

![df_describe](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%201/images/df_describe.png?raw=true)
Gambar 3. Statistik Data pada Dataset **Red Wine Quality**

![heatmap_corr](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%201/images/Heatmap_corr.png?raw=true)
Gambar 4. *Heatmap Correlation* pada Dataset **Red Wine Quality**

![df_skew_before](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%201/images/df_skew(before).png?raw=true)
Gambar 5. Persebaran Data pada Dataset **Red Wine Quality**

Dari informasi-informasi tersebut, dapat disimpulkan bahwa:
- Tidak ada data yang **NULL** (total data = 1599)
- Terdapat 12 fitur / kolom dan sesuai keterangan pemilik dataset bahwa fitur `quality` akan menjadi targetnya
- Semua data berbentuk numerik (*float* dan *int*)
- Fitur `quality` ternyata hanya bernilai dari 3 hingga 8 (tidak sesuai keterangan pemilik dataset bahwa bernilai 0 hingga 10)
- Fitur `pH` bernilai 2.74 hingga 4.01 sehingga menunjukkan bahwa dataset ini sesuai kondisi riil (*wine* bersifat asam)
- Fitur `residual sugar` dan `free sulfur dioxide` memiliki korelasi yang rendah terhadap fitur target, yakni `quality`
- Terdapat beberapa data yang memiliki *outlier* dan *skewed* seperti pada fitur `chlorides`, `residual sugar`, `sulphates`, `total sulfur dioxide`, dan `free sulfur dioxide`

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
Gambar 6. Histogram pada Dataset **Red Wine Quality**
![boxplot_before](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%201/images/Boxplot(before).png?raw=true)
Gambar 7. *Boxplot* pada Dataset **Red Wine Quality**

3. Mengatasi data yang *skewed* atau memiliki banyak *outlier* dengan menggunakan metode IQR (*interquartile range*). Berikut ini rumus dari metode IQR dan implementasinya pada bahasa pemrograman *python*.
- $$\text{Batas Bawah} = Q1 - (1.5 \times \text{IQR})$$
- $$\text{Batas Atas} = Q3 + (1.5 \times \text{IQR})$$
![IQR_formula](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%201/images/IQR_Formula.png?raw=true)
Gambar 8. Rumus Metode IQR
Hal ini bertujuan untuk membersihkan data agar model *machine learning* dapat memahami data yang ada dengan lebih baik. Berikut ini hasil penerapan metode IQR pada dataset **Red Wine Quality**.
![df_info_after](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%201/images/df_info(after).png?raw=true)
Gambar 9. Jumlah Data dan Tipe Data pada Dataset Red Wine Quality Setelah Menerapkan Metode IQR
![df_skew_after](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%201/images/df_skew(after).png?raw=true)
Gambar 10. Persebaran Data pada Dataset **Red Wine Quality** Setelah Menerapkan Metode IQR
![histogram_after](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%201/images/Histogram(after).png?raw=true)
Gambar 11. Histogram pada Dataset **Red Wine Quality** Setelah Menerapkan Metode IQR
![boxplot_after](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%201/images/Boxplot(before).png?raw=true)
Gambar 12. *Boxplot* pada Dataset **Red Wine Quality** Setelah Menerapkan Metode IQR
4. Melakukan seleksi fitur dengan mempertimbangkan korelasi dari fitur `quality` dengan fitur lainnya. Korelasi fitur yang dilakukan menggunakan metode *Pearson Correlation* dengan menjalankan fungsi `df.corr()` yang juga divisualisasikan menggunakan *library* **seaborn**. Tujuan seleksi fitur ini untuk mengurangi dimensi data (agar tidak terjadi *overfitting*) dan meningkatkan akurasi model agar dapat memahami data yang relevan terhadap fitur target (yakni `quality`)
5. Melakukan proses *labeling* pada fitur `quality` untuk mempermudah klasifikasi.

Setelah proses tersebut selesai, dataset akan dipisah menjadi 2 bagian, yakni bagian **train** dan **valid** dengan perbandingkan data train:valid = 7:3. Adapun pembagian data tersebut menggunakan *library* **sklearn**. Tujuan pembagian data tersebut untuk menguji akurasi model yang ada.


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

**Hasil Hyperparameter Tuning => Tidak ada perubahan akurasi yang signifikan (kurang dari 1%)**

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

**Hasil Hyperparameter Tuning => Terjadi kenaikan akurasi sebesar 2%**

### Model ANN
**Kelebihan**
- Implementasi yang mudah
- Fleksibel untuk mengatur parameter

**Kekurangan**
- Rentan mengalami *overfitting*
- Biaya komputasi cenderung mahal (membutuhkan gpu pada beberapa kasus)

### Model Terbaik
Setelah dilakukan beberapa kali percobaan, didapatkan model terbaik untuk klasifikasi kualitas *wine* pada dataset ini adalah model **XGBoost** yang telah di-*hyperparameter Tuning* dengan akurasi sekitar 77%. 

## Evaluation
Metrik evaluasi yang digunakan pada penelitian ini adalah menggunakan fungsi dari *library* **Scikit-learn** (**sklearn**), yakni `classification_report`. Alasan menggunakan fungsi tersebut adalah untuk mempermudah mendapatkan informasi terkait metrik seperti **precision**, **recall**, **F1 score**, dan **akurasi**. Berikut ini penjelasan mengenai keempat metrik tersebut:
- **Precision** = mengukur seberapa banyak dari item yang diprediksi sebagai positif benar-benar benar positif.![Precision](https://miro.medium.com/v2/resize:fit:828/format:webp/1*qMqJuCi_0fGr0rwu9uP27w.png)
- **Recall** = mengukur seberapa banyak dari semua item yang benar-benar positif yang berhasil diidentifikasi oleh model. ![Recall](https://miro.medium.com/v2/resize:fit:828/format:webp/1*mQQ7vz3zfFOL9vJdt9LDqg.png)
- **F1 score** =  rata-rata harmonik dari precision dan recall. ![F1 score](https://miro.medium.com/v2/resize:fit:898/1*7tC4-fUHtcffvXGcGTJJtg.png)
- **Akurasi** = rasio dari jumlah prediksi yang benar (*True Positives* dan *True Negatives*) dibagi dengan jumlah total prediksi. ![Akurasi](https://miro.medium.com/v2/resize:fit:828/format:webp/1*ZOm7TWPX97sZpnjpS_YT4w.png)

Untuk mengetahui hasil prediksi secara detail, penulis juga menampilkan `confusion_matrix` pada **Jupyter Notebook**. Untuk lebih lengkapnya, dapat dilihat hasilnya pada Jupyter Notebook.

**---Sekian, Terima Kasih---**