import cv2 as cv

image = cv.imread('python/static/uploads/470.jpg')

blue, green, red = cv.split(image)

# convert image to grayscale
img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# apply binary thresholding
ret, thresh = cv.threshold(img_gray, 150, 255, cv.THRESH_BINARY)

contours, hierarchy = cv.findContours(image=thresh, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)

image_copy = image.copy()
cv.drawContours(image=img_gray, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)


cv.imshow('None approximation', image_copy)
cv.waitKey(0)
cv.imwrite('contours_none_image1.jpg', image_copy)
cv.destroyAllWindows()