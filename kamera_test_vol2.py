from pathlib import Path
from collections import deque
import csv
import math
import time

import cv2
import joblib
import mediapipe as mp  # Görüntüdeki eli tespit ediyor./  Hazır Hand Landmarker modelini çalıştırıyor./ MediaPipe tek elindeki 21 noktanin iskelet baglantilarini okur.


MODEL_YOLU = Path(__file__).parent / "models" / "hand_landmarker.task"
SINIFLANDIRICI_YOLU = (
    Path(__file__).parent / "models" / "sayi_siniflandirici.joblib"
)
VERI_YOLU = Path(__file__).parent / "veri" / "el_verileri.csv"
MINIMUM_GUVEN = 0.70
TAHMIN_PENCERESI = 8

# MediaPipe elindeki 21 noktanin iskelet baglantilari.
EL_BAGLANTILARI = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def el_iskeletini_ciz(kare, el_noktalari):
    yukseklik, genislik = kare.shape[:2]
    piksel_noktalari = [
        (int(nokta.x * genislik), int(nokta.y * yukseklik))
        for nokta in el_noktalari
    ]

    for baslangic, bitis in EL_BAGLANTILARI:
        cv2.line(
            kare,
            piksel_noktalari[baslangic],
            piksel_noktalari[bitis],
            (0, 255, 0),
            2,
        )

    for x, y in piksel_noktalari:
        cv2.circle(kare, (x, y), 4, (0, 0, 255), -1)


def el_noktalarini_normalize_et(el_noktalari):
    """Eli bilege gore ortalar ve boyutunu standartlastirir."""
    bilek = el_noktalari[0]
    goreli_noktalar = [
        (nokta.x - bilek.x, nokta.y - bilek.y, nokta.z - bilek.z)
        for nokta in el_noktalari
    ]

    olcek = max(
        math.sqrt(x * x + y * y + z * z)
        for x, y, z in goreli_noktalar
    )
    if olcek == 0:
        return None

    return [
        koordinat / olcek
        for nokta in goreli_noktalar
        for koordinat in nokta
    ]


def veri_basligini_hazirla():
    VERI_YOLU.parent.mkdir(parents=True, exist_ok=True)
    if VERI_YOLU.exists() and VERI_YOLU.stat().st_size > 0:
        return

    baslik = ["etiket"]
    for nokta_no in range(21):
        baslik.extend(
            [f"x{nokta_no}", f"y{nokta_no}", f"z{nokta_no}"]
        )

    with VERI_YOLU.open("w", newline="", encoding="utf-8") as dosya:
        csv.writer(dosya).writerow(baslik)


def etiket_sayaclarini_oku():
    sayaclar = {etiket: 0 for etiket in range(1, 6)}
    if not VERI_YOLU.exists():
        return sayaclar

    with VERI_YOLU.open("r", newline="", encoding="utf-8") as dosya:
        for satir in csv.DictReader(dosya):
            try:
                etiket = int(satir["etiket"])
                if etiket in sayaclar:
                    sayaclar[etiket] += 1
            except (KeyError, TypeError, ValueError):
                continue

    return sayaclar


def ornek_kaydet(etiket, el_noktalari):
    koordinatlar = el_noktalarini_normalize_et(el_noktalari)
    if koordinatlar is None:
        return False

    with VERI_YOLU.open("a", newline="", encoding="utf-8") as dosya:
        csv.writer(dosya).writerow([etiket, *koordinatlar])
    return True


def siniflandiriciyi_yukle():
    if not SINIFLANDIRICI_YOLU.is_file():
        raise FileNotFoundError(
            f"Sayi siniflandirma modeli bulunamadi: {SINIFLANDIRICI_YOLU}"
        )

    paket = joblib.load(SINIFLANDIRICI_YOLU)
    if not isinstance(paket, dict) or "model" not in paket:
        raise ValueError("Siniflandirma modeli gecersiz bir yapida.")

    siniflandirici = paket["model"]
    metadata = paket.get("metadata", {})
    if metadata.get("ozellik_sayisi") != 63:
        raise ValueError("Siniflandirma modeli 63 el koordinati beklemiyor.")
    if not hasattr(siniflandirici, "predict_proba"):
        raise ValueError("Siniflandirma modeli guven olasiligi uretmiyor.")

    return siniflandirici, metadata


def canli_tahmin_yap(siniflandirici, olasilik_gecmisi, el_noktalari):
    koordinatlar = el_noktalarini_normalize_et(el_noktalari)
    if koordinatlar is None:
        return None

    olasiliklar = siniflandirici.predict_proba([koordinatlar])[0]
    olasilik_gecmisi.append(olasiliklar)
    ortalama_olasiliklar = [
        sum(kare_olasiliklari[i] for kare_olasiliklari in olasilik_gecmisi)
        / len(olasilik_gecmisi)
        for i in range(len(olasiliklar))
    ]
    en_iyi_index = max(
        range(len(ortalama_olasiliklar)),
        key=ortalama_olasiliklar.__getitem__,
    )
    tahmin = int(siniflandirici.classes_[en_iyi_index])
    guven = float(ortalama_olasiliklar[en_iyi_index])
    return tahmin, guven


def main():
    if not MODEL_YOLU.is_file():
        raise FileNotFoundError(
            f"El takip modeli bulunamadi: {MODEL_YOLU}"
        )

    siniflandirici, model_metadata = siniflandiriciyi_yukle()
    print(
        f"Siniflandirici yuklendi: {model_metadata.get('model_adi', 'bilinmiyor')}"
    )

    veri_basligini_hazirla()
    sayaclar = etiket_sayaclarini_oku()
    aktif_etiket = None
    olasilik_gecmisi = deque(maxlen=TAHMIN_PENCERESI)

    ayarlar = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_YOLU)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    kamera = cv2.VideoCapture(0)
    if not kamera.isOpened():
        raise RuntimeError("Kamera acilamadi.")

    print("1-5: etiketi sec | SPACE: bir ornek kaydet | q: cikis")
    baslangic_zamani = time.perf_counter()

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(ayarlar) as eller:
            while True:
                basarili_mi, kare = kamera.read()
                if not basarili_mi:
                    print("Kameradan goruntu alinamadi.")
                    break

                # OpenCV BGR, MediaPipe ise RGB renk sirasi kullanir.
                rgb_kare = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
                mp_goruntu = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_kare,
                )

                zaman_ms = int(
                    (time.perf_counter() - baslangic_zamani) * 1000
                )
                sonuc = eller.detect_for_video(mp_goruntu, zaman_ms)

                for el_noktalari in sonuc.hand_landmarks:
                    el_iskeletini_ciz(kare, el_noktalari)

                if len(sonuc.hand_landmarks) == 1:
                    tahmin_sonucu = canli_tahmin_yap(
                        siniflandirici,
                        olasilik_gecmisi,
                        sonuc.hand_landmarks[0],
                    )
                    if tahmin_sonucu is None:
                        tahmin_metni = "Tahmin: hesaplanamadi"
                        tahmin_rengi = (0, 0, 255)
                    else:
                        tahmin, guven = tahmin_sonucu
                        if guven >= MINIMUM_GUVEN:
                            tahmin_metni = (
                                f"Tahmin: {tahmin} | Guven: %{guven * 100:.0f}"
                            )
                            tahmin_rengi = (0, 255, 0)
                        else:
                            tahmin_metni = (
                                f"Tahmin: Belirsiz ({tahmin}, %{guven * 100:.0f})"
                            )
                            tahmin_rengi = (0, 165, 255)
                elif len(sonuc.hand_landmarks) > 1:
                    olasilik_gecmisi.clear()
                    tahmin_metni = "Tahmin: yalnizca bir el gosterin"
                    tahmin_rengi = (0, 165, 255)
                else:
                    olasilik_gecmisi.clear()
                    tahmin_metni = "Tahmin: el bekleniyor"
                    tahmin_rengi = (200, 200, 200)

                etiket_metni = (
                    str(aktif_etiket) if aktif_etiket is not None else "secilmedi"
                )
                adet = sayaclar.get(aktif_etiket, 0)
                cv2.putText(
                    kare,
                    f"Etiket: {etiket_metni} | Kayit: {adet}/300",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    kare,
                    "1-5: etiket | SPACE: kaydet | Q: cikis",
                    (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    kare,
                    tahmin_metni,
                    (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    tahmin_rengi,
                    2,
                )

                cv2.imshow("El Takibi", kare)
                tus = cv2.waitKey(1) & 0xFF

                if ord("1") <= tus <= ord("5"):
                    aktif_etiket = int(chr(tus))
                    print(f"Aktif etiket: {aktif_etiket}")
                elif tus == ord(" "):
                    if aktif_etiket is None:
                        print("Once 1-5 tuslarindan bir etiket secin.")
                    elif len(sonuc.hand_landmarks) == 0:
                        print("Kayit yapilmadi: goruntude el bulunamadi.")
                    elif len(sonuc.hand_landmarks) > 1:
                        print("Kayit yapilmadi: kadrajda yalnizca bir el olmali.")
                    elif ornek_kaydet(
                        aktif_etiket, sonuc.hand_landmarks[0]
                    ):
                        sayaclar[aktif_etiket] += 1
                        print(
                            f"Etiket {aktif_etiket}: "
                            f"{sayaclar[aktif_etiket]}/300 kaydedildi."
                        )
                elif tus == ord("q"):
                    break
    finally:
        kamera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
