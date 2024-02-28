# Laporan Proyek 2 Machine Learning Mahir - Andika Rahman Teja

## Domain Proyek

*Anime* merupakan salah satu produk budaya pop Jepang yang berbasis animasi. *Anime* mulai dibawa masuk ke Amerika Serikat sejak era pendudukan Amerika Serikat atas Jepang dan mengalami perkembangan pasar yang cukup pesat [1]. *Anime* Jepang sangat digemari oleh masyarakat di dunia, termasuk di Indonesia. Hal ini sesuai dengan data yang disajikan oleh *The Association of Japanese Animations* (AJA) mengenai jumlah kontrak industri animasi Jepang di berbagai negara sepanjang tahun 2019, yang menunjukkan bahwa Indonesia menempati peringkat kelima di Asia Tenggara [2]. Kontrak tersebut meliputi lisensi penayangan anime serta royalti *Original Soundtrack Theme Song Anime* [3]. 

Antusiasme masyarakat yang tinggi terhadap penayangan *anime* Jepang melalui layanan *streaming* mengakibatkan industri di sektor tersebut banyak mengalami keuntungan. Hal ini dibuktikan pada tahun 2019, industri *anime* mendapatkan keuntungan sebesar 68.5 miliar Yen untuk pendapatan domestik. Keberadaan layanan *streaming*, baik berbayar maupun gratis, semakin mempermudah penonton untuk menikmati *anime* [4]. Untuk dapat meningkatkan kepuasaan penonton *anime*, beberapa distributor dan penyiar *anime* memerlukan data preferensi penggunanya untuk dapat diolah menjadi sebuah sistem rekomendasi yang menampilkan *anime* yang sesuai dengan pengguna tersebut [5]. Oleh karena itu, pada penelitian ini, akan membuat sebuah sistem rekomendasi *anime* tersebut.


## Business Understanding

### Rumusan Masalah
- Bagaimana cara membuat sistem rekomendasi *anime* berdasarkan genre yang mirip?
- Bagaimana cara membuat sistem rekomendasi *anime* berdasarkan riwayat *anime* yang pernah ditonton dan diberi rating serta faktor penilaian pengguna lain?

### Tujuan
- Untuk mengetahui cara membuat sistem rekomendasi *anime* berdasarkan genre yang mirip
- Untuk mengetahui cara membuat sistem rekomendasi *anime* berdasarkan riwayat *anime* yang pernah ditonton dan diberi rating serta faktor penilaian pengguna lain

### Dampak Penelitian terhadap Bisnis
- Membantu distributor dan penyiar *anime* dalam menyesuaikan *anime* yang akan disiarkan berdasarkan rekomendasi dari para pengguna
- Membantu distributor dan penyiar *anime* dalam memonetisasi penanyangan *anime* kepada konsumen dengan menawarkan rekomendasi *anime* yang sesuai preferensi konsumen tersebut dalam bentuk iklan, konten berbayar, penjualan *merchandise*, dan sebagainya.
- Membantu pengguna dalam menemukan konten *anime* berdasarkan preferensinya dan pertimbangan rating *anime* yang diberikan oleh pengguna lain

### Solusi yang ditawarkan
- Menggunakan model **Content Based Filtering** untuk membuat sistem rekomendasi *anime* berdasarkan genre yang mirip
- Menggunakan model **Collaborative Filtering** untuk membuat membuat sistem rekomendasi *anime* berdasarkan penilaian pengguna lain


## Data Understanding

Adapun dataset yang digunakan untuk penelitian ini adalah dataset yang bersifat *open source* di [Kaggle](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset). Dikarenakan faktor keterbatasan komputasi pada penelitian ini, dataset yang akan digunakan dari *website* tersebut hanya 2 data, yakni `anime-dataset-2023.csv` dan `users-score-2023.csv`. Dengan sedikit penyesuaian data (menyesuaikan jumlah fitur / kolom dan jumlah data), berikut informasi yang terdapat pada kedua data tersebut:

### `anime-dataset-2023.csv`

|Fitur|Jumlah Data Tidak NULL|Tipe Data|
|---|---|---|
|anime_id|19976|*int64*|
|Name|19976|*object*|
|Genres|19976|*object*|
|Other name|19976|*object*|
|Score|19976|*object*|
|Type|19976|*object*|

Tabel 1. *Overview* Data **anime-dataset-2023**

**Keterangan fitur dari Tabel 1**:
- `anime_id` => ID unik untuk setiap *anime*
- `Name` => Judul *anime*
- `Genres` => Genre pada *anime* tersebut
- `Other name` => Judul *anime* dalam bahasa lain (bahasa Jepang, Cina, atau Korea)
- `Score` => Skor atau rating *anime* tersebut
- `Type` => Tipe penayangan *anime* (serial TV, film, OVA, dan sebagainya)

**Kesimpulan informasi dari data anime-dataset-2023**:
- Terdapat 6 fitur / kolom dengan masing-masing fitur memiliki 19976 data
- Data tersebut tidak memiliki nilai NULL
- ID *anime* tidak urut karena terdapat angka yang terlewati (misal ID 2, 3, dan 4 tidak ada pada data)
- Hampir semua *anime* pada data tersebut memiliki lebih dari 1 genre
- Fitur `Score` memiliki rentang nilai dari 1-10

### `users-score-2023.csv`

|Fitur|Jumlah Data Tidak NULL|Tipe Data|
|---|---|---|
|user_id|200000|*int64*|
|Username|199997|*object*|
|anime_id|200000|*int64*|
|Anime Title|200000|*object*|
|rating|200000|*int64*|

Tabel 2. *Overview* Data **users-score-2023**

**Keterangan fitur dari Tabel 2**:
- `user_id` => ID unik untuk setiap *user* atau pengguna
- `Username` => Nama dari *user*
- `anime_id` => ID unik untuk setiap *anime*
- `Anime Title` => Judul *anime*
- `rating` => Rating yang diberikan oleh *user* untuk *anime* tersebut

**Kesimpulan informasi dari data users-score-2023**:
- Terdapat 5 fitur / kolom dengan masing-masing fitur memiliki 199997 data
- Data tersebut memiliki nilai NULL sebanyak 3
- ID *user* tidak urut
- ID *anime* tidak urut karena terdapat angka yang terlewati 
- Fitur `anime_id` pada data ini memiliki representasi yang sama dengan fitur `anime_id` pada data **anime-dataset-2023**
- Fitur `Anime Title` pada data ini memiliki nilai yang sama dengan fitur `Name` pada data **anime-dataset-2023** jika berdasarkan ID nya
- Fitur `rating` memiliki rentang nilai dari 1-10


## Data Preparation

### `anime-dataset-2023.csv`

1. Melakukan pemeriksaan nilai unik pada fitur `Score` karena seharusnya skor memiliki tipe data *int* atau *float*. Ditemukan fakta bahwa terdapat nilai UNKNOWN pada fitur tersebut. Nilai UNKNOWN tersebut juga ditemukan pada fitur `Other name` dan `Type`.
2. Pada fitur `Score`, terdapat 6037 data yang bernilai UNKNOWN (30% dari keseluruhan data). Untuk menangani nilai UNKNOWN pada fitur ini, dilakukan pengubahan nilai UNKNOWN menjadi nilai 0 (nol) dengan asumsi tidak ada *user* yang memberikan penilaian *anime* tersebut. Selain itu, alasan lain mengubah nilai UNKNOWN tersebut adalah untuk tidak kehilangan banyak informasi jika data dengan nilai UNKNOWN di-*drop*.
3. Pada fitur `Other name` dan `Type` tidak dilakukan penanganan apapun dikarenakan jumlah nilai UNKNOWN pada kedua fitur sangat kecil (kurang dari 1% dari keseluruhan data) dan kedua fitur tidak mempengaruhi sistem rekomendasi yang akan dibuat.
4. Berikut ini visualisasi data untuk fitur `Score` setelah mengubah nilai UNKNOWN:

![score_histogram](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%202/images/Score_Histogram.png?raw=true)

Gambar 1. Histogram Fitur `Score`

Pada Gambar 1, data terdistribusi secara normal (dengan mengabaikan nilai 0 sebagai representasi nilai UNKNOWN)

5. Memisahkan fitur dengan menggunakan fitur `anime_id`, `Name`, dan `Genres` sebagai fitur latih untuk model.
6. Melakukan pemisahan genre *anime* dengan melakukan **One-Hot Encoding**. Setelah dilakukan proses tersebut, diketahui jumlah jenis genre yang ada pada data adalah sebanyak 21 jenis.

### `users-score-2023.csv`

1. Melakukan *drop* data pada data yang memiliki nilai NULL. Hal ini dikarenakan jumlah data yang NULL sangat kecil (kurang dari 1%) dan membuat data menjadi lebih logis (tidak mungkin terdapat *user* yang memiliki ID namun tidak memiliki nama sehingga kondisi tersebut diragukan kebenarannya).
2. Berikut ini visualisasi fitur `rating` setelah dilakukan *drop* data:

![rating_histogram](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%202/images/Rating_Histogram.png?raw=true)

Gambar 2. Histogram Fitur `rating`

Pada Gambar 2, data terdistribusi secara normal.

3. Menggabungkan `df_user` yang menyimpan data `user_score_2023.csv` dengan `df_anime_cut` yang menyimpan data `anime_id` dan `Genres` dari data `anime-dataset-2023.csv`. Penggabungan ini menggunakan fungsi `merge` pada *library* **pandas** berdasarkan nilai `anime_id` dari kedua variabel yang bersesuaian. Tujuan penggabungan ini untuk membantu mengukur tingkat akurasi model dalam memberikan rekomendasi nantinya.
4. Melakukan proses *encoding* dan *mapping* pada fitur `user_id` dan `anime_id`. Hal ini bertujuan untuk membantu model dalam melakukan rekomendasi kepada seorang *user* nantinya.
5. Memisahkan fitur dengan menggunakan fitur `user` (hasil *mapping* nilai *encoding* fitur `user_id`) dan fitur `anime` (hasil *mapping* nilai *encoding* fitur `anime_id`) sebagai fitur *input* model. Sedangkan fitur `rating` akan digunakan sebagai fitur *output* model.
6. Melakukan standarisasi dengan menggunakan `StandardScaler()` dari *library* **sklearn** untuk fitur `rating` atau fitur *output*. Hal ini bertujuan untuk memudahkan model dalam melakukan proses *training*.
7. Membagi data *input* dan *output* dengan menggunakan `train_test_split()` dari *library* **sklearn** dengan rasio data latih : data test sebesar 3:1.


## Modeling

### Content Based Filtering

Pada penelitian ini, dilakukan teknik **Content Based Filtering** berdasarkan genre *anime* dan menggunakan data `anime-dataset-2023.csv`. Model yang digunakan pada teknik ini adalah `cosine_similarity()` dari *library* **sklearn**. Cara kerja dari model ini adalah mencari nilai kedekatan antar item. Semakin mirip kedua item, maka nilainya semakin mendekati 1. Sedangkan semakin tidak mirip kedua item, maka nilainya semakin mendekati 0. Adapun rumus yang digunakan pada model ini adalah sebagai berikut:

$$\text{{Cosine Similarity}}(A, B) = \frac{{A \cdot B}}{{\|A\| \|B\|}}$$

dengan keterangan:
- A dan B => array bilangan dari data atau item
- A . B => hasil perkalian dot item A dan item B
- |A| |B| => norma Euclidean dari masing-masing item A dan item B

Dengan mengetahui nilai kedekatan tersebut, maka model akan memberikan sebanyak N rekomendasi *anime* yang sesuai berdasarkan nilai kedekatan tertinggi dengan *anime* tersebut.

**Kelebihan**
- Perhitungan yang relatif cepat dan efisien
- Memiliki kemampuan untuk memahami relasi semantik antar item dengan baik

**Kekurangan**
- Rentan terhadap data yang *noise*
- Tidak dapat memperhatikan urutan dan konteks khusus yang dapat mempengaruhi sistem rekomendasi

### Collaborative Filtering

Pada penelitian ini, dilakukan teknik **Collaborative Filtering** berdasarkan rating yang diberikan *user* lain. Model yang digunakan pada teknik ini adalah implementasi *deep learning* dengan menggunakan *layer* **Embedding**. Adapun arsitektur model *deep learning* yang diimplementasikan pada *class* `RecommenderNet()` adalah sebagai berikut:

|Urutan *Layer*|Tipe *Layer*|Ukuran *Output*|Jumlah Parameter|
|---|---|---|---|
|1|Embedding (user_vector)|multiple|4646304|
|2|Embedding (user_bias)|multiple|96798|
|3|Embedding (anime_vector)|multiple|393024|
|4|Embedding (anime_bias)|multiple|8188|
|5|Flatten|multiple|0|
|6|Dense|multiple|128|
|7|Dense|multiple|65|

Tabel 3. Arsitektur Model untuk **Collaborative Filtering**

Berdasarkan arsitektur model tersebut, berikut cara kerja model `RecommenderNet()`:
1. Data `user` akan memasuki *layer* `Embedding` (sesuai pada *layer* 1) untuk memetakan nilai vektor dari data tersebut
2. Data `user` juga akan memasuki *layer* `Embedding` (sesuai pada *layer* 2) untuk mendapatkan nilai bias untuk perkalian dot pada operasi selanjutnya
3. Data `anime` akan memasuki *layer* `Embedding` (sesuai pada *layer* 3) untuk memetakan nilai vektor dari data tersebut
4. Data `anime` juga akan memasuki *layer* `Embedding` (sesuai pada *layer* 4) untuk mendapatkan nilai bias untuk perkalian dot pada operasi selanjutnya
5. Melakukan operasi perkalian dot dari nilai hasil *layer* 1 dan 3 yang kemudian dilakukan penjumlahan nilai bias hasil *layer* 2 dan 4
6. Melakukan operasi `Flatten` untuk memapatkan dimensi data sehingga dapat memasuki *layer* selanjutnya
7. Hasil dari *layer* `Flatten` dimasukkan ke *layer* `Dense` dengan fungsi aktivasi `relu`
8. Hasil dari *layer* `Dense` sebelumnya akan dimasukkan ke *layer* `Dense` dengan fungsi aktivasi `sigmoid`
9. *Output* dari layer terakhir adalah prediksi rekomendasi untuk *pair* antara *user* dan *anime*

Pada model ini juga menerapkan beberapa *callback* seperti `ModelCheckpoint`, `LearningRateScheduler`, dan `EarlyStopping`. Hal ini bertujuan agar model dapat menemukan titik optimum dengan baik dan mencegah *Exploding Gradient* pada model.

Hasil akhir dari model ini adalah memberikan rekomendasi sebanyak N rekomendasi *anime* yang paling sesuai dengan riwayat *anime* yang ditonton dan diberi rating oleh penonton.

**Kelebihan**
- Mampu menangani data yang besar dan kompleks
- Mampu merepresentasikan keterkaitan antar item dengan baik

**Kekurangan**
- Rentan mengalami *overfitting*
- Cenderung memiliki model yang kompleks sehingga diperlukan biaya komputasi yang besar


## Evaluation

### Content Based Filtering

Model akan menyarankan 10 *anime* yang paling sesuai dengan *anime* yang telah dipilih sebelumnya berdasarkan kesamaan genre. Sebagai contoh, akan dipilih *anime* dengan judul **Re:Zero kara Hajimeru Isekai Seikatsu** dengan genre `Drama`, `Fantasy`, dan `Suspense`. Berikut 10 rekomendasi *anime* yang diberikan oleh model:

|Nomor|Judul *Anime*|Genre|
|---|---|---|
|1|Re:Zero kara Hajimeru Isekai Seikatsu 2nd Season|Drama, Fantasy, Suspense|
|2|Re:Zero kara Hajimeru Isekai Seikatsu - Hyouketsu no Kizuna|Drama, Fantasy, Suspense|
|3|Re:Zero kara Hajimeru Isekai Seikatsu 3rd Season|Drama, Fantasy, Suspense|
|4|Re:Zero kara Hajimeru Isekai Seikatsu 2nd Season Part 2|Drama, Fantasy, Suspense|
|5|Shigofumi: Sore kara|Drama, Fantasy, Suspense|
|6|Shigofumi|Drama, Fantasy, Suspense|
|7|Narutaru: Mukuro Naru Hoshi Tama Taru Ko|Drama, Suspense|
|8|Escape from Tsuki no Uragawa Zoo|Drama, Fantasy|
|9|Violet Evergarden Gaiden: Eien to Jidou Shuki Ningyou|Drama, Fantasy|
|10|Wonder Egg Priority|Drama, Fantasy|

Tabel 4. Hasil Rekomendasi Model **Content Based Filtering**

Untuk dapat mengetahui akurasi model **Content Based Filtering** dapat menggunakan metrik evaluasi berupa **Precision Content Based Filtering**. Berikut ini formula untuk metrik tersebut:

$$\text{Precision} = \frac{\text{Jumlah item relevan yang direkomendasikan}}{\text{Jumlah total item yang direkomendasikan}}$$

Berdasarkan metrik tersebut, **akurasi model adalah 100%** dengan alasan:
- Seluruh *anime* dengan `Genres` berupa `Drama, Fantasy, Suspense` terdapat pada rekomendasi yang disarankan oleh model
- Karena `Genres` dengan tipe `Drama, Fantasy, Suspense` hanya terdapat 7 film (termasuk yang *anime* yang dicari), maka sisanya akan menyesuaikan ketiga genre tersebut.
- $$\text{Precision} = \frac{10}{10} = 1$$

Dengan akurasi model tersebut, diharapkan dapat membantu *user* dalam menemukan konten *anime* yang mirip berdasarkan genre dari *anime* yang ditonton sebelumnya.

### Collaborative Filtering

Berikut ini hasil *training* model `RecommenderNet()`:

![RMSE_model](https://github.com/AndikaRT421/Dicoding-ML-Terapan/blob/master/Proyek%202/images/RMSE_Model.png?raw=true)

Gambar 3. *Root Mean Square Error* Model

Pada Gambar 3, model semula mempelajari data dengan baik. Namun dari *epoch* 8 hingga 25, model mengalami kondisi *overfitting*. Hal ini dibuktikan dari nilai *Root Mean Square Error* (RMSE) pada data *train* semakin membaik dan pada data *test* semakin memburuk. Kondisi ini dapat terjadi jika model yang digunakan terlalu kompleks untuk data pada penelitian ini. Untuk mengatasinya, dapat memberikan *layer* `BatchNormalization` atau `Regularization` ataupun `Dropout`.

Model akan menyarankan 10 *anime* yang paling sesuai dengan *user* berdasarkan *anime* yang pernah ditonton dan diberi rating serta genre *anime* yang menyesuaikan dan menggunakan data `anime-dataset-2023.csv` dan `users-score-2023.csv`. Sebagai contoh, akan dipilih *user* bernama Caramelito dengan ID 375717. Berikut ini riwayat *anime* yang pernah ditonton dan diberi rating oleh *user* tersebut:

|ID *Anime*|Judul *Anime*|Rating|Genre|
|---|---|---|---|
|1575|Code Geass: Hangyaku no Lelouch|7|Action, Award Winning, Drama, Sci-Fi|
|5081|Bakemonogatari|4|Mystery, Romance, Supernatural|
|1482|D.Gray-man|9|Action, Adventure, Fantasy|
|3731|Itazura na Kiss|9|Comedy, Romance|

Tabel 5. Riwayat *Anime* yang Pernah Ditonton dan Diberi Rating

Selanjutnya, model memberikan 10 rekomendasi *anime* yang sesuai dengan riwayat *anime* yang pernah ditonton dan diberi rating oleh *user* sebagai berikut:

|Nomor|Judul *Anime*|Genre|
|---|---|---|
|1|Gintama|Action, Comedy, Sci-Fi|
|2|Hunter x Hunter (2011)|Action, Adventure, Fantasy|
|3|Cowboy Bebop|Action, Award Winning, Sci-Fi|
|4|Clannad: After Story|Drama, Romance, Supernatural|
|5|Code Geass: Hangyaku no Lelouch R2|Action, Award Winning, Drama, Sci-Fi|
|6|Fullmetal Alchemist: Brotherhood|Action, Adventure, Drama, Fantasy|
|7|Steins;Gate|Drama, Sci-Fi, Suspense|
|8|Howl no Ugoku Shiro|Adventure, Award Winning, Drama, Fantasy, Romance|
|9|Great Teacher Onizuka|Comedy|
|10|Rurouni Kenshin: Meiji Kenkaku Romantan - Tsuioku-hen|Action, Drama, Romance|

Tabel 6. Hasil Rekomendasi Model **Collaborative Filtering**

Berdasarkan hasil rekomendasi di atas menunjukkan bahwa **model memiliki akurasi yang sangat baik**. Hal ini terbukti dari genre *anime* yang pernah ditonton oleh *user* sesuai dengan genre *anime* yang disarankan oleh model.

Dengan akurasi model tersebut, diharapkan dapat membantu *user* dalam menemukan konten anime yang mirip berdasarkan riwayat *anime* yang pernah ditonton dan diberi rating sebelumnya serta rekomendasi dari *user* lain.


# Referensi
1. A. P. B. Aji, "Peningkatan Popularitas Anime Jepang di Pasar Amerika Serikat di Era Pandemi Covid-19", Universitas Gadjah Mada, Yogyakarta, 2022.
2. H. Masuda, "Anime Industry Report 2019", *The Association of Japanese Animations*, Tokyo, 2019
3. I. Aisyah, "ANIME DAN GAYA HIDUP MAHASISWA (Studi pada Mahasiswa yang Tergabung dalam Komunitas Japan Freak UIN Jakarta)", Universitas Islam Negeri Syarif Hidayatullah, Jakarta, 2019.
4. A. Rahardini, "PENGARUH TERPAAN ANIME HAIKYU!! TERHADAP KEPUTUSAN PEMBELIAN SEPATU ASICS", Universitas Atma Jaya, Yogyakarta, 2023.
5. Z. Wang, X. Yu, N. Feng, and Z. Wang, “An improved collaborative movie recommendation system using Computational Intelligence,” Journal of Visual Languages &amp; Computing, vol. 25, no. 6, pp. 667–675, 2014.

**---Sekian, Terima Kasih---**