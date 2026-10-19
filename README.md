# Studi Kasus: Sensor Fusion Lab, Optik vs Radar vs Gabungan untuk Mengenali Penyebab Deforestasi

Berdasarkan kuliah praktisi **"Attributing Direct Drivers to Deforestation Alerts with Satellite Imagery and AI"** oleh **Bart Slagter** (Wageningen University & Research), pertemuan ke-6.

Aplikasi web kecil (Flask) yang melatih tiga model klasifikasi pada citra **sintetis** (optik saja, radar saja, gabungan), lalu memperlihatkan bagaimana akurasi masing-masing berubah ketika **tutupan awan** dinaikkan lewat slider.
Tampilannya mengikuti gaya situs gitingest.

## Hubungan dengan materi

| Poin materi | Implementasi |
|---|---|
| Sentinel-2 (optik) memberi detail tetapi terhalang awan | Piksel awan menggantikan nilai optik; akurasi model optik turun saat awan bertambah |
| Sentinel-1 (radar) menembus awan tetapi lebih sulit diinterpretasi | Citra radar tidak terpengaruh awan, tetapi diberi derau bintik (*speckle*) yang besar |
| Data optik dan radar digabung agar saling melengkapi | Mode "Gabungan" memakai fitur dari kedua sensor sekaligus |
| Klasifikasi penyebab: pertanian, pertambangan, penebangan, kebakaran | Lima kelas sintetis dengan bentuk berbeda (persegi, gumpalan, petak tersebar, pita sungai) |
| Kebingungan antara kebakaran dan banjir | *Confusion matrix* per mode; keduanya dibuat gelap di citra optik sehingga mudah tertukar |
| Pemantauan bulanan pada resolusi 10 m | Tidak disimulasikan; hanya satu "citra" 16 x 16 piksel per sampel |

## Cara kerja singkat

1. Tiap sampel adalah patch 16 x 16 piksel: latar hutan dengan satu area gangguan sesuai kelasnya, plus derau.
2. Optik punya 3 pita (merah, hijau, NIR), radar punya 2 (VV, VH).
3. Fitur per pita: rata-rata, simpangan baku, minimum, maksimum.
4. Model: Random Forest (scikit-learn), satu per mode. Data latih 150 patch per kelas dengan tutupan awan acak 0 sampai 90%.
5. Data uji 40 patch per kelas (200 total), dievaluasi ulang tiap slider digeser.

Contoh hasil satu kali jalan pada awan 70%: optik sekitar 69%, radar sekitar 92%, gabungan sekitar 99%.
Angka bisa berbeda sedikit jika pengaturan acak diubah.

## Cara menjalankan

```bash
pip install -r requirements.txt
python app.py
```

Buka `http://127.0.0.1:5000` di browser. Saat start, model dilatih dulu sehingga butuh sekitar 5 sampai 10 detik.

Isi folder:

```
app.py               # data sintetis, pelatihan model, API /api/run
templates/index.html # tampilan (HTML, CSS, JS dalam satu file)
requirements.txt
```

## Catatan penting

- **Semua citra sintetis.** Data Sentinel-1/2 asli tidak dipakai karena berukuran besar dan butuh akun. Hasilnya mendemokan konsep, bukan klaim performa di dunia nyata.
- **Bukan CNN.** Materi memakai *Convolutional Neural Network* pada tumpukan citra. Di sini dipakai Random Forest pada ringkasan statistik per pita agar ringan dan cepat dijalankan.
- Pola kebingungan kebakaran vs banjir dan kemenangan radar atas optik di bawah awan tebal **sengaja dibentuk** lewat parameter sintetis. Ini menunjukkan alasan penggabungan sensor, bukan temuan empiris.
- Satu patch hanya berisi satu penyebab, sedangkan di dunia nyata penyebab saling terkait (*driver co-occurrence*) dan perlu data deret waktu.
- Model hanya dilatih sampai awan 90%, sehingga pada 100% awan hasil mode gabungan ikut turun.

## Pengembangan lanjutan

Ganti model dengan CNN (PyTorch) pada patch yang sama, tambah dimensi waktu agar keterkaitan antarpenyebab bisa dilihat, uji dengan potongan Sentinel asli, dan tambah contoh berlabel kebakaran dan banjir seperti yang dilakukan pada materi untuk mengurangi kebingungan.
