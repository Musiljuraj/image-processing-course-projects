import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("WebAgg")


cv.namedWindow("zero_img", 0)

# 1. Vytvoření černé matice (obrazu)
# h (výška), w (šířka), c (kanály - BGR)
# Je vytvořena matice o rozměrech 500x300 pixelů se 3 barevnými kanály.
# Datový typ je uint8 (hodnoty 0-255).
zero_img = np.zeros(shape=(500, 300, 3), dtype=np.uint8)
print(f"Rozměry obrazu: {zero_img.shape}")

cv.imshow("zero_img", zero_img)
cv.waitKey(0)

# 2. Změna barvy jednoho pixelu
# Pixel na souřadnicích [50, 50] je nastaven na bílou barvu (255, 255, 255).
zero_img[50,50] = (255, 255, 255)

cv.imshow("zero_img", zero_img)
cv.waitKey(0)

# 3. Ukázka slicingu (řezů) v Pythonu
# Je vypsána část seznamu od indexu 3 do konce.
lst = [1, 2, 3, 4, 5, 6, 7, 8]
print(f"Slice seznamu: {lst[3:]}")

# 4. Obarvení celého řádku
# Celý 50. řádek matice je nastaven na červenou barvu (0, 0, 255 v BGR).
zero_img[50, :] = (0, 0, 255)

cv.imshow("zero_img", zero_img)
cv.waitKey(0)

# 5. Ruční průchod a obarvení pixelů (pomalé)
# Je provedena iterace přes všechny pixely a 50. řádek je nastaven na zelenou.
# Tento způsob je v Pythonu neefektivní oproti vektorizovaným operacím (viz výše).
h, w, c = zero_img.shape

for x in range(0, w):
    for y in range(0, h):
        if y == 50:
            zero_img[y, x] = (0, 255, 0)    

cv.imshow("zero_img", zero_img)
cv.waitKey(0)

# 6. Obarvení celého obrazu
# Všechny pixely v obraze jsou nastaveny na červenou barvu.
zero_img[:, :] = (0, 0, 255)

cv.imshow("zero_img", zero_img)
cv.waitKey(0)

# 7. Výběr a obarvení horizontálního pruhu
# Řádky 0 až 10 (exkluzivně) jsou nastaveny na žlutou barvu (0, 255, 255).
zero_img[0:10, :] = (0, 255, 255)

cv.imshow("zero_img", zero_img)
cv.waitKey(0)

# 8. Výběr a obarvení vertikálního pruhu
# Sloupce 0 až 10 jsou nastaveny na žlutou barvu.
# Tímto dojde k přepsání předchozí barvy v oblasti průniku.
zero_img[:, 0:10] = (0, 255, 255)

cv.imshow("zero_img", zero_img)
cv.waitKey(0)

# 9. Výběr obdélníkové oblasti (ROI - Region of Interest)
# Je vybrána oblast od řádku 0 do 10 a sloupce 0 do 10.
# Tato oblast je nastavena na modrou barvu (255, 0, 0).
zero_img[0:10, 0:10] = (255, 0, 0)

cv.imshow("zero_img", zero_img)
cv.waitKey(0)

# 10. Kopírování oblasti do nového obrazu
# Vybraná oblast (ROI) je zkopírována do nové proměnné `block`.
# Metoda .copy() zajišťuje, že se vytvoří hluboká kopie dat, nikoliv jen pohled.
block = zero_img[0:10, 0:10].copy()
print(f"Rozměry bloku: {block.shape}")

cv.imshow("block", block)
cv.imshow("zero_img", zero_img)
cv.waitKey(0)

# 11. Horizontální spojení obrazů
# Jsou načteny dva stejné obrazy. Flag 1 znamená načtení v barevném režimu.
image_hc1 = cv.imread("img-2.png", 1)
image_hc2 = cv.imread("img-2.png", 1)

# Pokud se obrazy nepodařilo načíst (jsou None), kód by vyhodil chybu.
if image_hc1 is not None and image_hc2 is not None:
    # Obrazy jsou spojeny vedle sebe (horizontálně). Musí mít stejnou výšku.
    img_hconcat = cv.hconcat([image_hc1, image_hc2])
    cv.imshow("hconcat", img_hconcat)
    cv.waitKey(0)
    cv.destroyAllWindows() 
else:
    print("Chyba při načítání obrázků pro hconcat.")

# 12. Zobrazení pomocí Matplotlib
# Matplotlib používá RGB, zatímco OpenCV používá BGR.
# Proto je nutné provést konverzi barevného prostoru.
image = cv.imread("img.png", 1)
image_2 = cv.imread("img-2.png", 1)

if image is not None and image_2 is not None:
    # Konverze z BGR do RGB pro správné zobrazení v Matplotlib
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_2 = cv.cvtColor(image_2, cv.COLOR_BGR2RGB)

    # Vytvoření dvou podgrafů (subplots) vedle se
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Stacked subplots')
    
    # Zobrazení prvního obrazu
    axs[0].imshow(image)
    axs[0].set_title("Obrázek 1")
    
    # Zobrazení druhého obrazu
    axs[1].imshow(image_2)
    axs[1].set_title("Obrázek 2")

    plt.show()
else:
    print("Chyba při načítání obrázků pro Matplotlib.")
