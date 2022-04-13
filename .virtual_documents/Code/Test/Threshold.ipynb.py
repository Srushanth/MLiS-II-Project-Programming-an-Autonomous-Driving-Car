import cv2
import numpy as np


img = cv2.imread('./10022.png')


cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


ret, thresh1 = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 250, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 250, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 250, 255, cv2.THRESH_TOZERO_INV)


cv2.imshow('Binary Threshold', thresh1)
cv2.waitKey(0)
cv2.imshow('Binary Threshold Inverted', thresh2)
cv2.waitKey(0)
cv2.imshow('Truncated Threshold', thresh3)
cv2.waitKey(0)
cv2.imshow('Set to 0', thresh4)
cv2.waitKey(0)
cv2.imshow('Set to 0 Inverted', thresh5)
cv2.waitKey(0)


cv2.destroyAllWindows()


# global thresholding
ret1, th1 = cv2.threshold(img, 253, 255, cv2.THRESH_BINARY)


cv2.imshow('img', th1)
cv2.waitKey(0)
cv2.destroyAllWindows()


lower_black = np.array([0, 0, 0], dtype = "uint16")
upper_black = np.array([250, 250, 250], dtype = "uint16")
black_mask = cv2.inRange(th1, lower_black, upper_black)
cv2.imshow('img', black_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


(h, w) = black_mask.shape


h, w


h/4


required = black_mask[int((3*h)/5):h, :w]


cv2.imshow('img1', th1)
cv2.waitKey(0)
cv2.imshow('img', required)
cv2.waitKey(0)
cv2.destroyAllWindows()



