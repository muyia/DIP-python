import cv2
import numpy as np
import matplotlib.pyplot as plt

# histogram equalization
def hist_equal(img, z_max=255):
    height, width, = img.shape
    S = height * width  * 1.
    img_equal = img.copy()
    sum_h = 0.
    for i in range(1, 255):
        ind = np.where(img == i)
        sum_h += len(img[ind])
        z_prime = sum_h / S * z_max
        img_equal[ind] = z_prime
    img_equal = img_equal.astype(np.uint8)
    return img_equal


img = cv2.imread("images/gamma.PNG",0)

# histogram normalization
img_equal = hist_equal(img)


# Display histogram
# plt.savefig("out_his.png")
# Save result
cv2.imshow("img",img)
cv2.imshow("result", img_equal)
plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.hist(img_equal.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.show()
# cv2.imwrite("img_equal.jpg", out)
cv2.waitKey(0)
cv2.destroyAllWindows()