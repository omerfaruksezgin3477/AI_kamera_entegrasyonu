"""El noktalarından 1-5 sayılarını sınıflandıran modeli eğitir.

Kamera ve internet CSV'lerini birleştirir, veriyi eğitim/doğrulama/test olarak
ayırır, üç farklı modeli karşılaştırır ve en iyi modeli diske kaydeder.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJE_YOLU = Path(__file__).resolve().parent
VERI_DOSYALARI = (
    PROJE_YOLU / "veri" / "el_verileri.csv",
    PROJE_YOLU / "veri" / "internet_el_verileri.csv",
)
MODEL_YOLU = PROJE_YOLU / "models" / "sayi_siniflandirici.joblib"
RAPOR_YOLU = PROJE_YOLU / "models" / "egitim_raporu.json"
ETIKETLER = (1, 2, 3, 4, 5)
RASTGELELIK_TOHUMU = 42


def ozellik_adlari():
    return [
        f"{eksen}{nokta_no}"
        for nokta_no in range(21)
        for eksen in ("x", "y", "z")
    ]


OZELLIKLER = ozellik_adlari()
BEKLENEN_BASLIK = ["etiket", *OZELLIKLER]


def dosya_ozeti(yol):
    ozet = hashlib.sha256()
    with yol.open("rb") as dosya:
        for parca in iter(lambda: dosya.read(1024 * 1024), b""):
            ozet.update(parca)
    return ozet.hexdigest()


def csv_oku(yol):
    if not yol.is_file():
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {yol}")

    satirlar = []
    with yol.open("r", newline="", encoding="utf-8") as dosya:
        okuyucu = csv.DictReader(dosya)
        if okuyucu.fieldnames != BEKLENEN_BASLIK:
            raise ValueError(
                f"Beklenmeyen CSV başlığı: {yol}\n"
                f"Beklenen {len(BEKLENEN_BASLIK)} sütun, "
                f"bulunan {len(okuyucu.fieldnames or [])} sütun."
            )

        for satir_no, satir in enumerate(okuyucu, start=2):
            try:
                etiket = int(satir["etiket"])
                koordinatlar = [float(satir[ad]) for ad in OZELLIKLER]
            except (KeyError, TypeError, ValueError) as hata:
                raise ValueError(
                    f"Geçersiz değer: {yol}, satır {satir_no}"
                ) from hata

            if etiket not in ETIKETLER:
                raise ValueError(
                    f"Geçersiz etiket {etiket}: {yol}, satır {satir_no}"
                )
            if not all(math.isfinite(deger) for deger in koordinatlar):
                raise ValueError(
                    f"Sonlu olmayan koordinat: {yol}, satır {satir_no}"
                )
            satirlar.append((etiket, koordinatlar))

    return satirlar


def verileri_hazirla():
    kaynak_sayilari = {}
    tum_satirlar = []
    for yol in VERI_DOSYALARI:
        satirlar = csv_oku(yol)
        kaynak_sayilari[yol.name] = len(satirlar)
        tum_satirlar.extend(satirlar)

    # Tamamen aynı kaydın iki kez öğrenilmesini ve farklı bölümlere sızmasını önle.
    benzersiz = {}
    for etiket, koordinatlar in tum_satirlar:
        anahtar = (etiket, *koordinatlar)
        benzersiz[anahtar] = (etiket, koordinatlar)

    tekrar_sayisi = len(tum_satirlar) - len(benzersiz)
    satirlar = list(benzersiz.values())
    x = np.asarray([koordinatlar for _, koordinatlar in satirlar], dtype=np.float64)
    y = np.asarray([etiket for etiket, _ in satirlar], dtype=np.int64)

    eksik_etiketler = sorted(set(ETIKETLER) - set(y.tolist()))
    if eksik_etiketler:
        raise ValueError(f"Veride eksik etiketler var: {eksik_etiketler}")

    return x, y, kaynak_sayilari, tekrar_sayisi


def modelleri_olustur():
    return {
        "KNN": Pipeline(
            [
                ("olcekle", StandardScaler()),
                (
                    "model",
                    KNeighborsClassifier(n_neighbors=7, weights="distance"),
                ),
            ]
        ),
        "SVM-RBF": Pipeline(
            [
                ("olcekle", StandardScaler()),
                (
                    "model",
                    CalibratedClassifierCV(
                        SVC(
                            kernel="rbf",
                            C=10.0,
                            gamma="scale",
                            class_weight="balanced",
                            random_state=RASTGELELIK_TOHUMU,
                        ),
                        method="sigmoid",
                        cv=5,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(  # random forest burada 500 tane random karar ağacı mevcut
            n_estimators=500,
            min_samples_leaf=2,                   # Bir ağacın en uç yaprağında en az iki eğitim örneği bulunmasını ister. Bu, ağacın tek bir örneği ezberlemesini azaltan ön budamadır.
            class_weight="balanced_subsample",
            n_jobs=-1,                            # 500 ağacı eğitirken bilgisayarın kullanılabilir işlemci çekirdeklerini kullanır. Eğitimin hızlı bitmesinin sebeplerinden biri budur.
            random_state=RASTGELELIK_TOHUMU,
        ),
    }


def metrikleri_hesapla(gercek, tahmin):
    return {
        "dogruluk": float(accuracy_score(gercek, tahmin)),
        "dengeli_dogruluk": float(balanced_accuracy_score(gercek, tahmin)),
        "makro_f1": float(f1_score(gercek, tahmin, average="macro")),
    }


def dagilim(y):
    sayac = Counter(int(etiket) for etiket in y)
    return {str(etiket): sayac.get(etiket, 0) for etiket in ETIKETLER}


def main():
    x, y, kaynak_sayilari, tekrar_sayisi = verileri_hazirla()

    # %15 test en başta ayrılır ve model seçimine kesinlikle katılmaz.
    x_gelistirme, x_test, y_gelistirme, y_test = train_test_split(
        x,
        y,
        test_size=0.15,
        random_state=RASTGELELIK_TOHUMU,
        stratify=y,
    )
    # Kalan %85'in yaklaşık %17.65'i toplam verinin %15'ine karşılık gelir.
    x_egitim, x_dogrulama, y_egitim, y_dogrulama = train_test_split(
        x_gelistirme,
        y_gelistirme,
        test_size=0.15 / 0.85,
        random_state=RASTGELELIK_TOHUMU,
        stratify=y_gelistirme,
    )

    print(f"Toplam benzersiz örnek: {len(y)}")
    print(
        f"Eğitim: {len(y_egitim)} | "
        f"Doğrulama: {len(y_dogrulama)} | Test: {len(y_test)}"
    )

    dogrulama_sonuclari = {}
    egitilmis_modeller = {}
    for ad, model in modelleri_olustur().items():
        model.fit(x_egitim, y_egitim)
        tahmin = model.predict(x_dogrulama)
        metrikler = metrikleri_hesapla(y_dogrulama, tahmin)
        dogrulama_sonuclari[ad] = metrikler
        egitilmis_modeller[ad] = model
        print(
            f"{ad}: makro F1={metrikler['makro_f1']:.3f}, "
            f"doğruluk={metrikler['dogruluk']:.3f}"
        )

    kazanan_ad = max(
        dogrulama_sonuclari,
        key=lambda ad: (
            dogrulama_sonuclari[ad]["makro_f1"],
            dogrulama_sonuclari[ad]["dengeli_dogruluk"],
        ),
    )

    # Kazanan mimariyi eğitim + doğrulama verisinin tamamıyla yeniden eğit.
    kazanan_model = clone(egitilmis_modeller[kazanan_ad])
    kazanan_model.fit(x_gelistirme, y_gelistirme)
    test_tahmini = kazanan_model.predict(x_test)
    test_metrikleri = metrikleri_hesapla(y_test, test_tahmini)
    test_raporu = classification_report(
        y_test,
        test_tahmini,
        labels=list(ETIKETLER),
        output_dict=True,
        zero_division=0,
    )
    karisiklik_matrisi = confusion_matrix(
        y_test, test_tahmini, labels=list(ETIKETLER)
    ).tolist()

    metadata = {
        "model_adi": kazanan_ad,
        "olusturulma_zamani_utc": datetime.now(timezone.utc).isoformat(),
        "etiketler": list(ETIKETLER),
        "ozellikler": OZELLIKLER,
        "ozellik_sayisi": len(OZELLIKLER),
        "rastgelelik_tohumu": RASTGELELIK_TOHUMU,
        "ornek_sayisi": int(len(y)),
        "silinen_tam_tekrar": tekrar_sayisi,
        "kaynak_sayilari": kaynak_sayilari,
        "tum_veri_dagilimi": dagilim(y),
        "bolum_dagilimlari": {
            "egitim": dagilim(y_egitim),
            "dogrulama": dagilim(y_dogrulama),
            "test": dagilim(y_test),
        },
        "dogrulama_sonuclari": dogrulama_sonuclari,
        "test_metrikleri": test_metrikleri,
        "test_siniflandirma_raporu": test_raporu,
        "karisiklik_matrisi": karisiklik_matrisi,
        "veri_dosyalari": {
            str(yol.relative_to(PROJE_YOLU)): {
                "sha256": dosya_ozeti(yol),
                "satir": kaynak_sayilari[yol.name],
            }
            for yol in VERI_DOSYALARI
        },
        "ortam": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "uyari": (
            "Rastgele bölme benzer kamera karelerini farklı kümelere dağıtabilir; "
            "test sonucu gerçek kullanıcı performansını olduğundan yüksek gösterebilir."
        ),
    }

    MODEL_YOLU.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": kazanan_model, "metadata": metadata},
        MODEL_YOLU,
        compress=3,
    )
    with RAPOR_YOLU.open("w", encoding="utf-8") as dosya:
        json.dump(metadata, dosya, ensure_ascii=False, indent=2)

    print(f"\nSeçilen model: {kazanan_ad}")
    print(
        f"Test makro F1={test_metrikleri['makro_f1']:.3f}, "
        f"test doğruluğu={test_metrikleri['dogruluk']:.3f}"
    )
    print("Karışıklık matrisi (satır=gerçek, sütun=tahmin):")
    print(np.asarray(karisiklik_matrisi))
    print(f"Model kaydedildi: {MODEL_YOLU}")
    print(f"Rapor kaydedildi: {RAPOR_YOLU}")


if __name__ == "__main__":
    main()
