import numpy as np
import cv2 as cv            
import matplotlib.pyplot as plt



def createImage():
    img = np.zeros((500,300,3), dtype=np.uint8)
    img[50,50]=(255,255,255)
    img[50,:] = (0,0,255)
    img[0:20,0:20] = (255,0,0)
    #resized = cv.resize(img, None, fx=0.5, fy=0.5)
    return img


def main():
    img = createImage()

    #roi = img[20:40,20:40].copy()
    #roi[:] = (0,255,0)
    cv.circle(img, (200, 200), 30, (0, 0, 255), -1)
    cv.putText(img, 'OpenCV', (10, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, lineType=cv.LINE_AA)

    cv.imwrite("img01.png", img)

    img = cv.imread('img01.png', 0)
    cv.imshow("img", img)
    cv.waitKey(0)
    cv.destroyAllWindows()



if __name__ == "__main__":    main()
