import numpy as np
import cv2


def myLBP(src):
    """
    Args:
        src: gray image
    """

    height = src.shape[0]
    width = src.shape[1]

    dst = src.copy()

    lbp_value = np.zeros((1, 8), dtype=np.uint8)

    neighbors = np.zeros((1, 8), dtype=np.uint8)

    for x in range(1, width-1):
        for y in range(1, height-1):
            center = src[y, x]

            neighbors[0, 0] = src[y-1, x-1]
            neighbors[0, 1] = src[y-1, x]
            neighbors[0, 2] = src[y-1, x+1]
            neighbors[0, 3] = src[y, x-1]
            neighbors[0, 4] = src[y, x+1]
            neighbors[0, 5] = src[y+1, x-1]
            neighbors[0, 6] = src[y+1, x]
            neighbors[0, 7] = src[y+1, x+1]

            for i in range(8):
                if neighbors[0, i] > center:
                    lbp_value[0, i] = 1
                else:
                    lbp_value[0, i] = 0

            # uint8 八位二进制数来存放0-1序列 巧妙！
            lbp = 0
            for i in range(8):
                lbp += lbp_value[0, i] * 2**i

            dst[y, x] = lbp

    return dst


img = cv2.imread("data/food-101/images/grilled_cheese_sandwich/36521.jpg", 0)
cv2.imshow("raw", img)
res = myLBP(img)
cv2.imshow("result", res)

print(img[:10, :10])
print(res[:10, :10])

cv2.waitKey(0)
cv2.destroyAllWindows()

