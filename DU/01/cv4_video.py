import cv2 as cv
import numpy as np

# 1. Inicializace snímání videa
# Parametr 0 obvykle značí výchozí webkameru.
# Pokud je připojených kamer více, lze použít index 1, 2, atd.
# Místo indexu kamery lze zadat i cestu k video souboru (např. 'video.mp4').
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Chyba: Nelze otevřít kameru.")
    exit()

print("Kamera úspěšně otevřena. Stiskněte 'q' pro ukončení.")

while True:
    # 2. Čtení jednotlivých snímků (rámců)
    # Metoda .read() vrací True/False (úspěch) a samotný snímek (frame).
    ret, frame = cap.read()

    # Pokud snímek nebyl načten (např. konec videa nebo odpojení kamery), cyklus se ukončí.
    if not ret:
        print("Konec streamu nebo chyba načítání snímku.")
        break

    # 3. Zpracování snímku
    # Zde provádíme operace na každém snímku stejně jako u statického obrazu.
    # Například převod na šedotón.
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 4. Zobrazení výsledku
    # Zobrazujeme původní barevný i upravený šedotónový obraz.
    cv.imshow('Kamera - Original', frame)
    cv.imshow('Kamera - Sedoton', gray)

    # 5. Ošetření ukončení klávesou
    # Funkce waitKey čeká na stisk klávesy po dobu 1 ms.
    # Pokud je stisknuta klávesa 'q' (kód znaku ord('q')), cyklus se přeruší.
    if cv.waitKey(1) == ord('q'):
        break

# 6. Uvolnění zdrojů
# Po ukončení práce je nutné uvolnit kameru a zavřít všechna okna.
cap.release()
cv.destroyAllWindows()
