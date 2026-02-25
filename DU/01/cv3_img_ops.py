import cv2 as cv
import numpy as np

# 1. Načtení obrazu ze souboru
# Parametr 1 (cv.IMREAD_COLOR) načte obraz barevně (BGR).
# Parametr 0 (cv.IMREAD_GRAYSCALE) by načetl obraz v odstínech šedi.
img = cv.imread('img.png', cv.IMREAD_COLOR)

if img is None:
    print("Chyba: Obrázek se nepodařilo načíst. Zkontrolujte cestu k souboru 'img.png'.")
else:
    print(f"Původní rozměry obrazu: {img.shape}")
    
    cv.imshow('Original', img)
    cv.waitKey(0)

    # 2. Změna velikosti obrazu (Resize)
    # Změna velikosti na polovinu původní šířky a výšky.
    # fx=0.5, fy=0.5 jsou faktory změny velikosti pro osu x a y.
    resized_img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
    print(f"Nové rozměry obrazu: {resized_img.shape}")

    cv.imshow('Resized', resized_img)
    cv.waitKey(0)

    # 3. Uložení obrazu do nového souboru
    # Upravený (zmenšený) obraz je uložen jako 'resized_img.jpg'.
    cv.imwrite('resized_img.jpg', resized_img)
    print("Zmenšený obraz byl uložen jako 'resized_img.jpg'.")

    # 4. Převod barevného prostoru (Color Conversion)
    # Převod z BGR do šedotónu (Grayscale).
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    cv.imshow('Gray', gray_img)
    cv.waitKey(0)

    # Převod z BGR do HSV (Hue, Saturation, Value).
    # HSV model je často vhodnější pro detekci barev.
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    cv.imshow('HSV', hsv_img)
    cv.waitKey(0)
    
    cv.destroyAllWindows()
