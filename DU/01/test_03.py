import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use("WebAgg")
from matplotlib import pyplot as plt

# #1 ==============================
# zero_img = np.zeros(shape=(500,500,3), dtype=np.uint8)
# print(f"Rozmery obrazu: {zero_img.shape}")
# #cv.imshow("zero_img", zero_img)
# #cv.waitKey(0)

# zero_img[:,:] = (0,0,255)
# cv.imshow("zero_img", zero_img)
# cv.waitKey(0)

# zero_img[0:10,:] = (255,255,0)
# cv.imshow("zero_img", zero_img)
# cv.waitKey(0)

# zero_img[10:20,10:20] = (0,255,0)
# cv.imshow("zero_img", zero_img)
# cv.waitKey(0)

# roi1 = zero_img[15:25,15:25].copy()
# cv.imshow("roi1", roi1)
# cv.imshow("zero_img", zero_img)
# cv.waitKey(0)

# cv.destroyAllWindows()
# #11 ====================================

# img_hc1 = cv.imread("img-2.png", 1)
# img_hc2 = cv.imread("img-2.png", 1)

# cv.imshow("img_hc1", img_hc1)
# cv.waitKey(0)

# if (img_hc1 is not None) and (img_hc1 is not None):
#     img_hconcat = cv.hconcat([img_hc1, img_hc2])
#     cv.imshow("img_hconcat", img_hconcat)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
# else:
#     print("Chyba pri nacitani obrazu pre hconcat")

# 12 ===========================================
img01 = cv.imread("img.png", 1)
img02 = cv.imread("img-2.png", 1)
#cv.imshow("img01", img01)
#cv.waitKey(0)

if (img01 is not None) and (img02 is not None):
    img01 = cv.cvtColor(img01, cv.COLOR_BGR2RGB)
    img02 = cv.cvtColor(img02, cv.COLOR_BGR2RGB)
    #cv.imshow("img01", img01)
    #cv.waitKey(0)
    fig, axs = plt.subplots(1,2)
    fig.suptitle('Stack subplots')

    axs[0].imshow(img01)
    axs[0].set_title("Image 1")

    axs[1].imshow(img02)
    axs[1].set_title("Image 2")

    plt.show()
else:
    print("Chyba pri nacitani obrazu img01")
