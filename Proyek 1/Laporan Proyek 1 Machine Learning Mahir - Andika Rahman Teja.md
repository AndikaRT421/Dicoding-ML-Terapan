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

### Dampak Bisnis
- 

### Solusi yang ditawarkan
- Menggunakan model *Adaptive Boosting* berupa `AdaBoostClassifier` beserta *hyperparameter tuning*-nya
- Menggunakan model *Gradient Boosting* berupa `XGBClassifier` beserta *hyperparameter tuning*-nya
- Menggunakan model *Artificial Neural Network* berupa model `sequential`


## Data Understanding

Adapun dataset yang digunakan untuk penelitian ini adalah dataset yang bersifat *open source* di [Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009). Selain itu, dataset tersebut juga tersedia di [UCI Machine Learning](https://archive.ics.uci.edu/dataset/186/wine+quality). Untuk visualisasi data, telah penulis cantumkan visualisasi histogram dan *boxplot* pada **Jupyter Notebook**.

### Variabel-Variabel Pada Dataset:
- `fixed acidity` =  jumlah asam yang tetap dalam *wine* setelah semua asam yang mudah menguap telah dihilangkan
- `volatile acidity` = jumlah asam yang mudah menguap dalam *wine*
- `citric acid` = komponen asam yang ditemukan dalam buah-buahan, termasuk anggur
- `residual sugar` = jumlah gula yang tersisa setelah proses fermentasi selesai
- `chlorides` = Kandungan garam klorida dalam *wine*
- `free sulfur dioxide` = sulfur dioksida dalam bentuk bebas yang terlarut dalam *wine*
- `total sulfur dioxide` = jumlah total sulfur dioksida dalam *wine*, termasuk yang terikat dengan senyawa lain
- `density` = massa *wine* per satuan volume pada suhu dan tekanan tertentu (kepadatan *wine*)
- `pH` = ukuran keasaman atau kebasaan pada *wine*
- `sulphates` = kandungan sulfur pada *wine*
- `alcohol` = kandungan alkohol pada *wine*
- `quality` [**Output Variabel**] = kualitas *wine*


## Data Preparation
Setelah dataset berhasil terbaca pada Jupyter Notebook melalui *library* **pandas** dan **numpy**, selanjutnya adalah melakukan proses EDA (*Exploratory Data Analysis*) dan *data cleaning*. Adapun EDA dan *data cleaning* yang dilakukan pada penelitian ini adalah sebagai berikut:
1. Menjalankan fungsi `df.info()` dan `df.describe()` untuk mengatahui jumlah data dan fitur / kolom data, memastikan apakah data memiliki nilai NULL, dan mengetahui tipe data tiap fitur. Hasilnya telah penulis cantumkan pada **Jupyter Notebook**
2. Melakukan visulisasi data untuk mengetahui persebaran dan kondisi dari data
3. Mengatasi data yang *skewed* atau memiliki banyak *outlier* dengan menggunakan metode IQR (*interquartile range*). Hal ini bertujuan untuk membersihkan data agar model *machine learning* dapat memahami data yang ada dengan lebih baik
4. Melakukan seleksi fitur dengan mempertimbangkan korelasi dari fitur `quality` dengan fitur lainnya. Korelasi fitur yang dilakukan menggunakan metode *Pearson Correlation* dengan menjalankan fungsi `df.corr()` yang juga divisualisasikan menggunakan *library* **seaborn**. Tujuan seleksi fitur ini untuk mengurangi dimensi data (agar tidak terjadi *overfitting*) dan meningkatkan akurasi model agar dapat memahami data yang relevan terhadap fitur target (yakni `quality`)
5. Melakukan proses *labeling* pada fitur `quality` untuk mempermudah klasifikasi.

Setelah proses tersebut selesai, dataset akan dipisah menjadi 2 bagian, yakni bagian **train** dan **valid** dengan perbandingkan data train:valid = 7:3. Adapun pembagian data tersebut menggunakan *library* **sklearn**. Tujuan pembagian data tersebut untuk menguji akurasi model yang ada.

**Untuk lebih lengkapnya, dapat dilihat prosesnya pada Jupyter Notebook**


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