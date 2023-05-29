# Gerekli Python kütüphanelerini ekle
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream, VideoStream
from imutils import face_utils
import argparse
import imutils
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Göz açıklık oranlarını tutmak için bir numpy array oluştur
ratioList = np.zeros(10)

#boş bir grafik oluştur
plt.ion()  # Etkileşimli modu etkinleştir
fig, ax = plt.subplots()
line, = ax.plot([], [])  # Boş bir çizgi oluştur

# Grafiği güncelle
def update_plot():
    ax.relim()  # Eksen sınırlarını yeniden hesapla
    ax.autoscale_view()  # Otomatik ölçeklendirmeyi etkinleştir
    line.set_data(range(len(ratioList)), ratioList)  # Veriyi güncelle
    ax.set_ylim(0, 50)  # Y ekseni sınırlarını ayarla
    fig.canvas.draw()  # Grafiği çiz

# Arka arkaya kaç adet düşük değer gelmesini belirle
EYE_AR_CONSEC_FRAMES = 2


COUNTER = 0 # Kare sayacı
TOTAL = 0 # Göz kırpma sayacı

# Göz açıklığı oranında ciddi bir düşüşü tespit etmek için kullanılan
# fonksiyon dizideki maksimum değerden minimum değer çıkarılır ve aralarındaki
# fark belirli bir sayının üstündeyse True döndürür
def detectBlink(ratioList):
    try:
        max_val = max(ratioList)
        index_max = np.where(np.array(ratioList) == max_val)[0][0]
        min_val = min(ratioList[:index_max])
        diff = max_val - min_val
    except:
        return False

    if diff > 10:
        return True
    else:
        return False

# Komut satırından alınabilecek argümanları belirle
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="veriseti dosya yolu")
ap.add_argument("-v", "--video", type=str, default="", help="Girdi olarak verilecek video kaynagi")
args = vars(ap.parse_args())


def eye_aspect_ratio(eye):
    # İki adet dikey göz sınırının arasındaki öklidyen mesafesini hesapla
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # İki adet yatay göz sınırının arasındaki öklidyen mesafesini hesapla
    C = dist.euclidean(eye[0], eye[3])

    # Göz açıklık oranını hesapla 
    ear = (A + B) / (2.0 * C) * 100
    return ear

# dlib yüz algılama ve yüz hatları tespit etme araçlarını başlat
print("[INFO] yüz sınır işaretleyici yükleniyor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["dataset"])

# sol ve sağ göz için yüz sınır işaretlerinin indekslerini al
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# video akışı iş parçacığını başlat
print("[INFO] video akış iş parçacığı başlatılıyor...")
if args["video"] == "webcam":
    vs = VideoStream(src=0).start()  # Webcam kullanımı için
    fileStream = False
else:
    vs = FileVideoStream(args["video"]).start()
    fileStream = True

# video akışındaki kareleri döngü ile işle
while True:
    # eğer bu bir dosya video akışıysa, buffer'daki daha fazla kare olup olmadığını kontrol etmeliyiz
    if fileStream and not vs.more():
        break

    # iş parçacığından video akışından kare al, yeniden boyutlandır ve siyah-beyaz kareye dönüştür
    # (kanalları)
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    frame = cv2.flip(frame, 1) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # siyah-beyaz kare içinde yüzleri tespit et
    rects = detector(gray, 0)

    # tespit edilen yüzler üzerinde döngü yap
    for rect in rects:
        # yüz bölgesi için yüz işaretlerini belirle, ardından
        # (x, y) koordinatlarını bir NumPy dizisine dönüştür
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # sol ve sağ göz koordinatlarını al, ardından
        # her iki göz için göz açıklık oranını hesapla
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # her iki göz için göz açıklık oranlarını birlikte ortala
        ear = (leftEAR + rightEAR) / 2.0

        # sol ve sağ göz için çizgilerin çevresini hesapla ve
        # her iki gözü görselleştir
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # göz açıklık oranınında ciddi bir düşüş olup olmadığını kontrol et
        ratioList = ratioList[:9]
        ratioList = np.append(ear, ratioList)
        print(ratioList)
        if detectBlink(ratioList):
            COUNTER += 1

        else:
            # gözler yeterli sayıda kapalıysa
            # toplam göz kırpması sayısını artır
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1

            # göz kırpması sayacını sıfırla
            COUNTER = 0

        # kare üzerinde toplam göz kırpması sayısını ve
        # kare için hesaplanan göz açıklık oranını yazdır
        cv2.putText(frame, "Sayac: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Oran: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        update_plot()

    # Ekrana getirframe = cv
    cv2.imshow("Frame", frame)
    plt.show()
    plt.pause(0.000000001)
    key = cv2.waitKey(1) & 0xFF

    # eğer "q" tuşuna basıldıysa döngüden çık
    if key == ord("q"):
        break

# Pencereleri kapat
cv2.destroyAllWindows()
vs.stop()
