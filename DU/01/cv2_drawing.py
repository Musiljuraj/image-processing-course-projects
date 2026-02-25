import cv2 as cv
import numpy as np

# Vytvoření černého obrazu o rozměrech 512x512 pixelů, 3 kanály (BGR)
img = np.zeros((512, 512, 3), np.uint8)

# 1. Vykreslení úsečky (Line)
# Parametry: obraz, počáteční bod, koncový bod, barva (BGR), tloušťka
# Je vykreslena modrá čára z levého horního do pravého dolního rohu.
cv.line(img, (0, 0), (511, 511), (255, 0, 0), 5)

# 2. Vykreslení obdélníku (Rectangle)
# Parametry: obraz, levý horní roh, pravý dolní roh, barva, tloušťka
# Je vykreslen zelený obdélník.
cv.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)

# 3. Vykreslení kružnice (Circle)
# Parametry: obraz, střed, poloměr, barva, tloušťka (-1 pro vyplnění)
# Je vykreslen červený plný kruh uvnitř předchozího obdélníku.
cv.circle(img, (447, 63), 63, (0, 0, 255), -1)

# 4. Vložení textu (PutText)
# Parametry: obraz, text, souřadnice, font, měřítko, barva, tloušťka, typ čáry
# Je vložen bílý text "OpenCV".
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv.LINE_AA)

# Zobrazení výsledného obrazu
cv.imshow("Drawing", img)
cv.waitKey(0)
cv.destroyAllWindows()
