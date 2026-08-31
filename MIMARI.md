# El Sayısı Tanıma Projesi Mimarisi

## Kısa özet

Bu proje, kameradan veya etiketli internet görsellerinden aldığı tek el
görüntüsünü önce MediaPipe ile 21 eklem noktasına dönüştürür. Noktalar bileğe
göre merkezlenip el boyutuna göre normalize edilir ve 63 sayısal özellik elde
edilir. KNN, SVM ve Random Forest bu özellikler üzerinde karşılaştırılır;
doğrulamada seçilen model diske kaydedilir. Kamera uygulaması kaydedilen modeli
yükler, son sekiz karenin olasılıklarını ortalar ve canlı `1-5` tahmini gösterir.

```text
Kamera / JPG
    -> OpenCV
    -> MediaPipe Hand Landmarker
    -> 21 x (x,y,z)
    -> normalizasyon
    -> 63 özellik
    -> Random Forest
    -> sayı + güven
```

Sistem statik el pozlarını tanır. El sallama gibi zamana yayılan hareketler için
kare dizisi ve ayrı bir zaman-serisi mimarisi gerekir.

## Ayrıntılı katmanlar

### 1. Görüntü edinme ve arayüz

`kamera_test_vol2.py`, OpenCV ile varsayılan kamerayı açar, kareleri okur,
iskeleti ve tahmin sonucunu görüntüye çizer, klavye girişlerini yönetir.

Girdi: BGR kamera karesi  
Çıktı: RGB MediaPipe görüntüsü ve kullanıcıya gösterilen işlenmiş kare

### 2. El algılama ve özellik çıkarma

`models/hand_landmarker.task`, önceden eğitilmiş MediaPipe modelidir. Görüntüde
elin yerini ve 21 eklem için `x, y, z` koordinatlarını çıkarır. Bu model sayıyı
bilmez; yalnızca el geometrisini üretir.

### 3. Ön işleme

`el_noktalarini_normalize_et`, bileği sıfır noktası yapar ve bütün koordinatları
eldeki en büyük bilek mesafesine böler. Böylece ekrandaki konum ve elin kameraya
uzaklığı daha az etkili olur. Dönüş açısı tamamen normalize edilmez.

Girdi: 21 ham nokta  
Çıktı: sabit sırada 63 kayan noktalı özellik

### 4. Veri katmanı

- `veri/el_verileri.csv`: kameradan elle toplanan örnekler
- `veri/internet_el_verileri.csv`: internet görsellerinden kabul edilen örnekler

Her satırın sözleşmesi aynıdır:

```text
etiket,x0,y0,z0,...,x20,y20,z20
```

CSV bu ölçekte yeterlidir. Eşzamanlı çok kullanıcılı veri toplama veya merkezi
bir API olmadığı için veritabanı gerekmez.

### 5. İnternet verisi kürasyonu

`internetten_veri_aktar.py`, daha önce indirilen görselleri MediaPipe ile aynı
63 özellik biçimine dönüştürür. Kullanıcı `E` ile kabul, `H` ile ret kararı verir.
JSON durum dosyası ilerlemeyi ve kurtarmayı sağlar; yalnız kabul edilen kayıtlar
CSV'ye girer. Bu katman otomatik etikete körü körüne güvenilmesini engeller.

### 6. Eğitim ve model seçimi

`modeli_egit.py`, iki CSV'yi doğrular, tam tekrarları kaldırır ve sınıf oranlarını
koruyarak yaklaşık `%70 eğitim / %15 doğrulama / %15 test` ayırır.

- KNN: en yakın yedi komşunun mesafe ağırlıklı oyunu
- SVM-RBF: ölçeklenmiş özelliklerde doğrusal olmayan sınıf sınırı
- Random Forest: 500 karar ağacının ortak oyu

Model doğrulama makro F1 değerine göre seçilir. Test kümesi seçime katılmaz.
Mevcut 830 örnekte Random Forest seçilmiş ve testte `%96` doğruluk, `%95,9`
makro F1 üretmiştir.

### 7. Model saklama

- `models/sayi_siniflandirici.joblib`: eğitilmiş sınıflandırıcı ve metadata
- `models/egitim_raporu.json`: metrikler, veri özetleri, sürümler ve karışıklık matrisi

Model yalnız güvenilen yerel dosyadan yüklenmelidir. Eğitim ve canlı kullanım
ortamındaki scikit-learn sürümleri aynı tutulmalıdır.

### 8. Canlı çıkarım

Kamera uygulaması tek el bulunduğunda aynı normalizasyonu uygular ve
`predict_proba` çağırır. Son sekiz karenin olasılıkları ortalanır. En yüksek
ortalama `%70` altındaysa sonuç `Belirsiz` gösterilir. İki el için tahmin
yapılmaz; çünkü eğitim satırları tek el geometrisidir.

## Dosya sorumlulukları

| Dosya | Sorumluluk |
|---|---|
| `kamera_test_vol2.py` | Kamera, MediaPipe, veri toplama ve canlı tahmin |
| `internetten_veri_aktar.py` | JPG -> 63 özellik, insan kontrollü kabul/ret |
| `modeli_egit.py` | Veri doğrulama, bölme, model karşılaştırma ve kaydetme |
| `hand_landmarker.task` | Önceden eğitilmiş el noktası çıkarma modeli |
| `sayi_siniflandirici.joblib` | Bu projede eğitilen `1-5` sınıflandırıcısı |
| `egitim_raporu.json` | Yeniden üretim ve değerlendirme raporu |

## Benzer bir projede izlenecek adımlar

1. **Problemi tanımla:** Statik poz mu, dinamik hareket mi; kaç sınıf var;
   bilinmeyen hareket nasıl ele alınacak?
2. **Etiket sözlüğünü sabitle:** Her sınıf için kabul edilen el biçimlerini resim
   ve metinle tanımla. Aynı sınıftaki alternatif biçimleri ayrıca işaretle.
3. **Veri sözleşmesini belirle:** Ham görüntü mü, landmark mı, özellik sırası ve
   birimleri ne olacak?
4. **Çeşitli veri topla:** Farklı kişi, el, açı, mesafe, ışık ve arka plan kullan;
   sınıf ve alt-biçim sayılarını dengele.
5. **Kaynak bilgisini koru:** Kişi/oturum/kaynak alanlarını sakla. Eğitim ve test
   aynı kişinin ardışık karelerini paylaşmamalı.
6. **Kalite kontrolü yap:** Yanlış etiket, MediaPipe hatası, tam tekrar ve aykırı
   örnekleri incele.
7. **Bölmeyi eğitimden önce yap:** Kişi veya oturuma göre eğitim/doğrulama/test
   ayır; test kümesine model seçerken bakma.
8. **Basit temel modelleri karşılaştır:** KNN, SVM ve Random Forest gibi uygun
   modelleri aynı metrik ve aynı bölümlerde ölç.
9. **Doğru metriği seç:** Sınıf dengesizliğinde yalnız doğruluk yerine makro F1,
   sınıf bazlı recall ve karışıklık matrisini kullan.
10. **Modeli ve tarifi sürümle:** Kod, veri özeti, bağımlılık sürümleri, eşik ve
    metrikleri modelle birlikte kaydet.
11. **Canlı çıkarımı sağlamlaştır:** Olasılık eşiği, `Belirsiz` sınıfı, zamansal
    yumuşatma ve hatalı giriş kontrolleri ekle.
12. **Gerçek kullanım testi yap:** Tamamen yeni kişi ve oturumlarda ölç; hatalı
    sınıflardan hedefli yeni veri topla ve modeli yeniden eğit.

## Mevcut `3` problemi

İlk model eğitildiği sırada toplam `3` sayısı yanıltıcı biçimde yeterli
görünmekteydi:

- Kamera ile kullanılan yerel `3` biçimi: 7 örnek
- İnternetten gelen alternatif `3` biçimi: 174 örnek

Yerel biçim, `3` sınıfının yalnızca yaklaşık `%3,9`udur. Kayıtlı model yedi yerel
örneği sınıf olarak doğru tahmin etse de `3` olasılıklarının ortalaması `%63,2`
ve medyanı `%68,6`dır; yedi örneğin dördü canlı `%70` eşiğinin altındadır.
İnternet `3` örneklerinde ortalama yaklaşık `%94,2`dir. Yalnız internet verisiyle
eğitilen model, yedi yerel `3` örneğinin beşini `4` olarak tahmin etmiştir.

İlk teşhisten sonra 34 yeni yerel örnek toplanmış ve kamera CSV'sindeki yerel
`3` sayısı 41'e yükselmiştir. Kaydedilmiş model bu yeni örneklerle henüz yeniden
eğitilmediğinden canlı davranışı değiştirmeleri için önce eğitim tekrarlanmalıdır.

Geleneksel `3` için ikinci internet kaynağı olarak Zenodo DOI
`10.5281/zenodo.3901659` seçilmiştir. Kaynak CC BY 4.0 lisanslıdır ve her sınıfta
2.400 train, 600 validation, 600 test görseli içerir. Yalnız `train/3` içinden
elle doğrulanan sınırlı bir alt küme eğitime alınmalı; kaynağın validation ve
test bölümleri ayrı tutulmalıdır.

Bu bulgular güven eşiğinden çok veri alanı ve alt-biçim dengesizliğini gösterir.
Önerilen çözüm:

1. `%70` eşiğini hemen düşürme; bu başka yanlış tahminleri gizleyebilir.
2. Yerel `3` biçiminden farklı oturum, açı, mesafe ve ışıklarda en az 100-150
   gerçek örnek topla.
3. Art arda neredeyse aynı kareler yerine eli her kayıtta yeniden konumlandır.
4. Alternatif iki `3` biçimini yaklaşık dengeli tut; ikisini de etiket `3` yap.
5. Yeni bir oturumdan 20-30 yerel `3` örneğini eğitime katmadan test için ayır.
6. Yeniden eğit ve özellikle `3` recall, güven dağılımı ve `3 -> 4` hatasını ölç.
