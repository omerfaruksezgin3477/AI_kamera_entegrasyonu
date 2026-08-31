# Python El Takibi ve Sayı Sınıflandırma

Python 3.13, OpenCV ve MediaPipe Hand Landmarker kullanarak web kamerasından
gerçek zamanlı el takibi yapan ve statik `1-5` el sayılarını öğrenmek için veri
üreten proje. MediaPipe her elden 21 eklem noktası çıkarır; proje bu noktaları
normalize ederek 63 sayısal özelliğe dönüştürür.

## Mimari

```text
Kamera veya internet görseli
              |
              v
OpenCV: görüntüyü oku, BGR -> RGB dönüştür
              |
              v
MediaPipe Hand Landmarker: 21 adet (x, y, z) noktası
              |
              v
Bileğe göre merkezleme + el boyutuna göre normalizasyon
              |
              v
63 özellik + 1 etiket içeren CSV
              |
              v
KNN / SVM / Random Forest karşılaştırması
              |
              v
Seçilen sayı sınıflandırma modeli
```

Bu sistemde iki ayrı model katmanı vardır:

1. `hand_landmarker.task`, Google MediaPipe tarafından önceden eğitilmiş el
   noktası çıkarma modelidir. Sayının ne olduğunu bilmez.
2. `sayi_siniflandirici.joblib`, bu projedeki etiketli 63 özellik üzerinden
   `1-5` sayılarını öğrenen modeldir.

Ham görüntü üzerinde büyük bir CNN eğitmek yerine el geometrisini kullandığımız
için veri ve işlemci ihtiyacı düşüktür. Arka planın etkisi de büyük ölçüde
MediaPipe katmanında elenir.

## Kurulum (Windows 11)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Calistirma

```powershell
python kamera_test_vol2.py
```

Programdan cikmak icin kamera penceresi acikken `q` tusuna basin.

## Veri toplama

Kamera uygulamasında `1-5` ile etiket seçilir, `Space` ile tek el örneği
`veri/el_verileri.csv` dosyasına kaydedilir. Her satırda bir etiket ve 21 nokta
için `x, y, z` olmak üzere 63 özellik bulunur.

İnternetten indirilen Sign Language Digits görselleri şu komutla elle incelenir:

```powershell
.\venv\Scripts\python.exe internetten_veri_aktar.py
```

İnceleme ekranında `E` kabul, `H` ret, `Q` kaydet ve çık anlamına gelir. Kabul
edilen örnekler `veri/internet_el_verileri.csv` dosyasına yazılır. Böylece klasör
etiketi doğru olsa bile projedeki el biçimine uymayan görseller eğitime girmez.

İnternet veri kaynağı: [Sign Language Digits Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset)
(Apache-2.0 lisansı).

## Model eğitimi

```powershell
.\venv\Scripts\python.exe modeli_egit.py
```

Program iki CSV'yi okur, tam tekrarları siler ve sınıf oranlarını koruyarak
veriyi yaklaşık `%70 eğitim / %15 doğrulama / %15 test` olarak ayırır.

- **KNN:** Yeni örneği kendisine en yakın eğitim örneklerinin oyuyla sınıflar.
  Mesafeye duyarlı olduğu için StandardScaler kullanılır.
- **SVM-RBF:** Sınıflar arasında doğrusal olmayan sınırlar oluşturur. Ölçekleme
  kullanır ve sınıf dengesini dikkate alır.
- **Random Forest:** Farklı veri alt kümelerinde çok sayıda karar ağacı kurar ve
  ağaçların ortak oyunu kullanır. Ek ölçekleme gerektirmez.

Model seçimi test kümesinde değil, doğrulama kümesindeki makro F1 değerine göre
yapılır. Seçilen mimari eğitim ve doğrulama verisiyle yeniden eğitildikten sonra
test kümesine yalnızca bir kez bakılır.

Mevcut 830 örnekte seçilen model Random Forest'tır:

- Test doğruluğu: `%96,0`
- Test makro F1: `%95,9`
- 125 test örneğinin 120'si doğru sınıflandırılmıştır.

Sonuçlar `models/egitim_raporu.json`, eğitilmiş model ise
`models/sayi_siniflandirici.joblib` içinde saklanır. Rapor veri dosyalarının
SHA-256 özetlerini ve kütüphane sürümlerini de içerir.

Bu test rastgele bölme kullandığı için birbirine benzeyen ardışık kamera kareleri
farklı bölümlere düşebilir. Bu nedenle `%96`, gerçek hayatta kesin başarı garantisi
değildir. Son değerlendirme farklı kişiler ve yeni kamera oturumlarıyla yapılmalıdır.

## Dosyalar

- `kamera_test_vol2.py`: Kamera ve el takip uygulamasi.
- `internetten_veri_aktar.py`: İnternet görsellerini elle kontrol edip CSV'ye dönüştürür.
- `modeli_egit.py`: Üç modeli karşılaştırır, seçilen modeli eğitir ve kaydeder.
- `models/hand_landmarker.task`: Resmi MediaPipe Hand Landmarker model paketi.
- `models/sayi_siniflandirici.joblib`: Eğitilmiş sayı sınıflandırıcısı.
- `models/egitim_raporu.json`: Metrikler, dağılımlar ve yeniden üretim bilgileri.
- `requirements.txt`: Gerekli Python kutuphaneleri.

Model, resmi [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
kaynagindan alinmistir.
