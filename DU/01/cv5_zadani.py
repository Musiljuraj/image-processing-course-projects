"""
PŘÍPRAVA PROSTŘEDÍ:
* Instalace balíku Ultralytics: 'pip install ultralytics' 
* https://github.com/ultralytics/ultralytics
* https://docs.ultralytics.com/modes/predict/#inference-arguments

Zadání na cvičení:

1. Inicializace detekčního modelu YOLO.
2. Načtení všech obrazových souborů z adresáře "bmw_100" a spuštění modelu YOLO na těchto souborech.
3. Implementace parametrů příkazové řádky: 
   - pro definici ID detekované třídy (např. 2 pro automobily).
   - pro velikost modelu
   - pro nazev vstupniho adresare
   - pro nazev vystupniho adresare
4. Uložení extrahovaných výřezů objektů do určené složky (např. 'car').
5. Vykreslení ohraničujících rámečků kolem objektu do původních obrazů (využití OpenCV). 
   Pokuste se o vykreslení ohraničujících rámečků (bounding boxes), které vrátí YOLO model pomocí funkce cv2.rectangle() případně informací o objektech + cv2.putText()
   - možnost nastavit barvu jako parametr příkazové řádky 
6. Vložení textové informace o celkovém počtu detekcí dané třídy do obrazu (např. levý dolní roh).

* ukázka spuštění s definicí barvy a specifické třídy:
python cv5_zadani.py --class_id 0 --color 0 0 255 --output_dir persons

0	osoba
1	jízdní kolo
2	osobní automobil
3	motocykl
5	autobus
"""

import cv2 as cv
from ultralytics import YOLO

# Inicializace modelu a cest
# ...

# Iterace souborů a čtení obrazů
# ...

# Inference, extrakce ROI a zápis na disk
# ...
