from pathlib import Path
import time

import cv2
import mediapipe as mp


MODEL_YOLU = Path(__file__).parent / "models" / "hand_landmarker.task"

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


def main():
    if not MODEL_YOLU.is_file():
        raise FileNotFoundError(
            f"El takip modeli bulunamadi: {MODEL_YOLU}"
        )

    ayarlar = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_YOLU)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    kamera = cv2.VideoCapture(0)
    if not kamera.isOpened():
        raise RuntimeError("Kamera acilamadi.")

    print("Kamera acildi. Programdan cikmak icin 'q' tusuna basin.")
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

                cv2.imshow("El Takibi", kare)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        kamera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
