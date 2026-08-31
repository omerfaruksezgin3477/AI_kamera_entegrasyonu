"""İnternetten indirilen el görsellerini inceleyerek CSV verisine dönüştürür.

Her görsel kullanıcıya gösterilir. Böylece klasör etiketi doğru olsa bile
bizim kullandığımız el biçimine uymayan örnekler veri setine eklenmez.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import cv2
import mediapipe as mp

from kamera_test_vol2 import (
    MODEL_YOLU,
    el_iskeletini_ciz,
    el_noktalarini_normalize_et,
)


PROJE_YOLU = Path(__file__).resolve().parent
VARSAYILAN_VERI_SETI = (
    PROJE_YOLU.parent / "Sign-Language-Digits-Dataset" / "Dataset"
)
CIKTI_YOLU = PROJE_YOLU / "veri" / "internet_el_verileri.csv"
DURUM_YOLU = PROJE_YOLU / "veri" / "internet_aktarim_durumu.json"
GORSEL_UZANTILARI = {".jpg", ".jpeg", ".png"}
ETIKETLER = range(1, 6)


def csv_basligi():
    baslik = ["etiket"]
    for nokta_no in range(21):
        baslik.extend(
            [f"x{nokta_no}", f"y{nokta_no}", f"z{nokta_no}"]
        )
    return baslik


def durum_yukle():
    gecici_yol = DURUM_YOLU.with_suffix(".json.tmp")
    adaylar = [
        yol
        for yol in (DURUM_YOLU, gecici_yol)
        if yol.exists() and yol.stat().st_size > 0
    ]
    if not adaylar:
        return {"surum": 1, "kayitlar": {}}

    # Bir önceki çalışmada Windows hedef dosyayı kilitlediyse daha yeni olan
    # geçici dosya son kararı içerir. Geçerli adaylar arasından en yenisini seç.
    gecerli_adaylar = []
    for yol in adaylar:
        try:
            with yol.open("r", encoding="utf-8") as dosya:
                aday_durum = json.load(dosya)
            if (
                aday_durum.get("surum") == 1
                and isinstance(aday_durum.get("kayitlar"), dict)
            ):
                gecerli_adaylar.append((yol.stat().st_mtime_ns, yol, aday_durum))
        except (OSError, json.JSONDecodeError):
            continue

    if not gecerli_adaylar:
        raise ValueError(f"Geçersiz aktarım durum dosyası: {DURUM_YOLU}")

    _, secilen_yol, durum = max(gecerli_adaylar, key=lambda aday: aday[0])
    if secilen_yol == gecici_yol:
        print("Önceki çalışmanın son kararı geçici dosyadan kurtarıldı.")
    return durum


def gecici_dosyayi_yerlestir(gecici_yol, hedef_yol):
    """Windows'un kısa süreli dosya kilitlerine karşı güvenli kayıt yapar."""
    son_hata = None
    for _ in range(20):
        try:
            gecici_yol.replace(hedef_yol)
            return
        except PermissionError as hata:
            son_hata = hata
            time.sleep(0.1)

    # Bazı Windows süreçleri yeniden adlandırmayı engellerken dosyanın üzerine
    # yazılmasına izin verir. Geçici dosya başarısızlık halinde yine korunur.
    try:
        with gecici_yol.open("rb") as kaynak, hedef_yol.open("wb") as hedef:
            shutil.copyfileobj(kaynak, hedef)
        gecici_yol.unlink()
    except PermissionError as hata:
        raise PermissionError(
            f"Kayıt dosyası Windows tarafından kilitli: {hedef_yol}. "
            f"Kurtarma kopyası burada: {gecici_yol}"
        ) from (son_hata or hata)


def atomik_json_yaz(durum):
    DURUM_YOLU.parent.mkdir(parents=True, exist_ok=True)
    gecici_yol = DURUM_YOLU.with_suffix(".json.tmp")
    with gecici_yol.open("w", encoding="utf-8") as dosya:
        json.dump(durum, dosya, ensure_ascii=False, indent=2)
    gecici_dosyayi_yerlestir(gecici_yol, DURUM_YOLU)


def csv_yeniden_olustur(durum):
    """Kabul edilen kayıtları durum dosyasından güvenle yeniden üretir."""
    CIKTI_YOLU.parent.mkdir(parents=True, exist_ok=True)
    gecici_yol = CIKTI_YOLU.with_suffix(".csv.tmp")

    kabul_edilenler = [
        (kaynak, kayit)
        for kaynak, kayit in durum["kayitlar"].items()
        if kayit["durum"] == "kabul"
    ]
    kabul_edilenler.sort(key=lambda oge: oge[0])

    with gecici_yol.open("w", newline="", encoding="utf-8") as dosya:
        yazici = csv.writer(dosya)
        yazici.writerow(csv_basligi())
        for _, kayit in kabul_edilenler:
            yazici.writerow([kayit["etiket"], *kayit["koordinatlar"]])

    gecici_dosyayi_yerlestir(gecici_yol, CIKTI_YOLU)


def gorselleri_bul(veri_seti_yolu):
    gorseller = []
    for etiket in ETIKETLER:
        etiket_yolu = veri_seti_yolu / str(etiket)
        if not etiket_yolu.is_dir():
            raise FileNotFoundError(f"Etiket klasörü bulunamadı: {etiket_yolu}")

        for gorsel_yolu in sorted(etiket_yolu.iterdir()):
            if gorsel_yolu.is_file() and gorsel_yolu.suffix.lower() in GORSEL_UZANTILARI:
                gorseller.append((etiket, gorsel_yolu))
    return gorseller


def inceleme_ekrani_hazirla(gorsel, el_noktalari, etiket, sira, toplam, kaynak):
    ekran = cv2.resize(gorsel, (600, 600), interpolation=cv2.INTER_CUBIC)
    el_iskeletini_ciz(ekran, el_noktalari)

    cv2.rectangle(ekran, (0, 0), (600, 90), (0, 0, 0), -1)
    cv2.putText(
        ekran,
        f"Etiket: {etiket} | {sira}/{toplam}",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        ekran,
        "E: kabul | H: reddet | Q: kaydet ve cik",
        (15, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        ekran,
        kaynak.name[:42],
        (15, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
    )
    return ekran


def ozet_yaz(durum):
    kayitlar = durum["kayitlar"].values()
    kabul = sum(kayit["durum"] == "kabul" for kayit in kayitlar)
    ret = sum(kayit["durum"] == "ret" for kayit in kayitlar)
    bulunamadi = sum(
        kayit["durum"] == "el_bulunamadi" for kayit in kayitlar
    )
    print(
        f"Durum: {kabul} kabul, {ret} ret, "
        f"{bulunamadi} el bulunamadı."
    )
    print(f"CSV: {CIKTI_YOLU}")


def main():
    ayrıştırıcı = argparse.ArgumentParser(
        description="İnternet görsellerini elle onaylayarak CSV'ye dönüştürür."
    )
    ayrıştırıcı.add_argument(
        "--dataset",
        type=Path,
        default=VARSAYILAN_VERI_SETI,
        help="İçinde 1,2,3,4,5 klasörleri bulunan Dataset yolu.",
    )
    ayrıştırıcı.add_argument(
        "--reddedilenleri-tekrar-incele",
        type=int,
        choices=ETIKETLER,
        metavar="ETIKET",
        help="Belirtilen etikette daha önce reddedilen görselleri yeniden açar.",
    )
    secenekler = ayrıştırıcı.parse_args()
    veri_seti_yolu = secenekler.dataset.expanduser().resolve()

    if not MODEL_YOLU.is_file():
        raise FileNotFoundError(f"MediaPipe modeli bulunamadı: {MODEL_YOLU}")
    if not veri_seti_yolu.is_dir():
        raise FileNotFoundError(f"İnternet veri seti bulunamadı: {veri_seti_yolu}")

    durum = durum_yukle()
    durum["veri_seti"] = str(veri_seti_yolu)
    tekrar_etiketi = secenekler.reddedilenleri_tekrar_incele
    if tekrar_etiketi is not None:
        tekrar_anahtarlari = [
            anahtar
            for anahtar, kayit in durum["kayitlar"].items()
            if kayit["etiket"] == tekrar_etiketi
            and kayit["durum"] == "ret"
        ]
        for anahtar in tekrar_anahtarlari:
            del durum["kayitlar"][anahtar]
        atomik_json_yaz(durum)
        print(
            f"Etiket {tekrar_etiketi}: {len(tekrar_anahtarlari)} "
            "reddedilmiş görsel yeniden incelemeye açıldı."
        )

    gorseller = gorselleri_bul(veri_seti_yolu)
    bekleyenler = [
        (etiket, yol)
        for etiket, yol in gorseller
        if f"{etiket}/{yol.name}" not in durum["kayitlar"]
    ]

    print(f"Toplam görsel: {len(gorseller)}")
    print(f"Daha önce incelenmiş: {len(gorseller) - len(bekleyenler)}")
    print(f"Bekleyen: {len(bekleyenler)}")
    if not bekleyenler:
        csv_yeniden_olustur(durum)
        ozet_yaz(durum)
        return

    ayarlar = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_YOLU)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(ayarlar) as eller:
            for sira, (etiket, gorsel_yolu) in enumerate(bekleyenler, start=1):
                kaynak_anahtari = f"{etiket}/{gorsel_yolu.name}"
                gorsel = cv2.imread(str(gorsel_yolu))
                if gorsel is None:
                    durum["kayitlar"][kaynak_anahtari] = {
                        "durum": "el_bulunamadi",
                        "etiket": etiket,
                        "neden": "görsel okunamadı",
                    }
                    atomik_json_yaz(durum)
                    continue

                buyutulmus = cv2.resize(
                    gorsel, (600, 600), interpolation=cv2.INTER_CUBIC
                )
                rgb_gorsel = cv2.cvtColor(buyutulmus, cv2.COLOR_BGR2RGB)
                mp_gorsuntu = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_gorsel,
                )
                sonuc = eller.detect(mp_gorsuntu)

                if len(sonuc.hand_landmarks) != 1:
                    durum["kayitlar"][kaynak_anahtari] = {
                        "durum": "el_bulunamadi",
                        "etiket": etiket,
                        "neden": "MediaPipe tek el bulamadı",
                    }
                    atomik_json_yaz(durum)
                    continue

                el_noktalari = sonuc.hand_landmarks[0]
                koordinatlar = el_noktalarini_normalize_et(el_noktalari)
                if koordinatlar is None:
                    durum["kayitlar"][kaynak_anahtari] = {
                        "durum": "el_bulunamadi",
                        "etiket": etiket,
                        "neden": "koordinatlar normalize edilemedi",
                    }
                    atomik_json_yaz(durum)
                    continue

                ekran = inceleme_ekrani_hazirla(
                    buyutulmus,
                    el_noktalari,
                    etiket,
                    sira,
                    len(bekleyenler),
                    gorsel_yolu,
                )
                cv2.imshow("İnternet Verisi İnceleme", ekran)

                while True:
                    tus = cv2.waitKey(0) & 0xFF
                    if tus in (ord("e"), ord("E")):
                        durum["kayitlar"][kaynak_anahtari] = {
                            "durum": "kabul",
                            "etiket": etiket,
                            "koordinatlar": koordinatlar,
                        }
                        break
                    if tus in (ord("h"), ord("H")):
                        durum["kayitlar"][kaynak_anahtari] = {
                            "durum": "ret",
                            "etiket": etiket,
                        }
                        break
                    if tus in (ord("q"), ord("Q")):
                        atomik_json_yaz(durum)
                        csv_yeniden_olustur(durum)
                        ozet_yaz(durum)
                        return

                atomik_json_yaz(durum)
    finally:
        cv2.destroyAllWindows()

    csv_yeniden_olustur(durum)
    ozet_yaz(durum)


if __name__ == "__main__":
    main()
