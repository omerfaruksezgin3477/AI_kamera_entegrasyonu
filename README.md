# Python El Takibi

Python 3.13, OpenCV ve MediaPipe Hand Landmarker kullanarak web kamerasindan
gercek zamanli el takibi yapan baslangic projesi. Program en fazla iki eli
algilar; her eldeki 21 eklem noktasini ve aralarindaki iskelet baglantilarini
kameradan gelen goruntunun uzerine cizer.

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

## Dosyalar

- `kamera_test_vol2.py`: Kamera ve el takip uygulamasi.
- `models/hand_landmarker.task`: Resmi MediaPipe Hand Landmarker model paketi.
- `requirements.txt`: Gerekli Python kutuphaneleri.

Model, resmi [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
kaynagindan alinmistir.
