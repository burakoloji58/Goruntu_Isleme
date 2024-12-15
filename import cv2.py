import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

# 1. Görüntüye Tuz ve Karabiber Gürültüsü Ekleme
# skimage kütüphanesi kullanılarak

def gurultu_ekle(resim, amount):
    gurultulu_resim = random_noise(resim, mode='s&p', amount=0.2)  #amount gürültü derecesi
    gurultulu_resim = np.array(255 * gurultulu_resim, dtype=np.uint8)  # normalize[0-1] eder ve eski haline çevirir
    return gurultulu_resim

# 2. Median Filtreyi Elle Uygulama
def medyan_filtreli(resim, filtre_boyu):
    kenar_filtresi = np.pad(resim, filtre_boyu // 2, mode='constant', constant_values=0) #kenar ekleme , eklenecek kenar sayısı, ne değer eklenecek, 0
    filtrelenmis_resim = np.zeros_like(resim) # hepsi 0 olan filtre üretir

    for i in range(resim.shape[0]):
        for j in range(resim.shape[1]):
            filtre = kenar_filtresi[i:i + filtre_boyu, j:j + filtre_boyu] # i den i+ kernela kadar yani y eksen , diğeri de x eksen
            filtrelenmis_resim[i, j] = np.median(filtre)

    return filtrelenmis_resim

# 3. Sobel Operatörüyle Kenar Algılama
def sobel_kenar(resim):
    sobel_x = cv2.Sobel(resim, cv2.CV_64F, 1, 0, ksize=3) # x yönü  gradiyant hesaplar
    sobel_y = cv2.Sobel(resim, cv2.CV_64F, 0, 1, ksize=3) # y yönü 

    gradient_buyuklugu = np.sqrt(sobel_x**2 + sobel_y**2) #Gradyan büyüklüğü, her pikselin kenar şiddetini ifade eder
    gradient_buyuklugu = np.uint8(255 * gradient_buyuklugu / np.max(gradient_buyuklugu)) #Belirli bir büyüklüğün altındaki değerler kaldırılarak kenarlar belirginleştirilir

    return gradient_buyuklugu


# Orijinal görüntüyü yükle ve gri tona çevir
resim = cv2.imread("ornek_goruntu.jpg", cv2.IMREAD_GRAYSCALE)

    # Gürültü ekle
gurultulu_resim = gurultu_ekle(resim, amount=0.05)

    # Median filtre uygula
filtre_boyu = 5  # Çekirdek boyutu
gurultusuz_resim = medyan_filtreli(gurultulu_resim, filtre_boyu)

    # Kenar algılama
kenar_algilanmis_resim = sobel_kenar(gurultusuz_resim)

    # Sonuçları görselleştir
plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.imshow(resim, cmap="gray")
plt.title("Orijinal Görüntü")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(gurultulu_resim, cmap="gray")
plt.title("Tuz ve Karabiber Gürültüsü")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(gurultusuz_resim, cmap="gray")
plt.title(f"Median Filtre ({filtre_boyu}x{filtre_boyu})")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(kenar_algilanmis_resim, cmap="gray")
plt.title("Kenar Algılama (Sobel)")
plt.axis("off")

plt.tight_layout()
plt.show()