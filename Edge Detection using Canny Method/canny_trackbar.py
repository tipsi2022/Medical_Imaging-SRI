import cv2


def nothing_thr(x):
    print(x)


def nothing_blur(x):
    print(x)


image_read1 = cv2.imread('xray.png', cv2.IMREAD_GRAYSCALE)
image_read2 = cv2.resize(image_read1, (512, 512))

win_name = 'Edges'

cv2.namedWindow(win_name)

cv2.createTrackbar('Threshold 1', win_name, 0, 255, nothing_thr)
cv2.createTrackbar('Threshold 2', win_name, 0, 255, nothing_thr)
cv2.createTrackbar('Blur', win_name, 0, 5, nothing_blur)

image = cv2.GaussianBlur(image_read2, (1, 1), cv2.BORDER_DEFAULT)
canny = cv2.Canny(image, 0, 0)

while True:
    cv2.imshow(win_name, canny)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    thr1 = cv2.getTrackbarPos('Threshold 1', win_name)
    thr2 = cv2.getTrackbarPos('Threshold 2', win_name)
    blur = cv2.getTrackbarPos('Blur', win_name)

    image = cv2.GaussianBlur(image_read2, ((2 * blur) + 1, (2 * blur) + 1), cv2.BORDER_DEFAULT)
    canny = cv2.Canny(image, thr1, thr2)

cv2.destroyAllWindows()
