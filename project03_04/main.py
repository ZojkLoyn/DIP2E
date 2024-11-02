# project 03_04
import numpy as np
# Filter 函数定义
def img_filter(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # 图像和卷积核的尺寸
    H, W = img.shape
    h, w = kernel.shape
    oH, oW = H-h+1, W-w+1
    # kernel翻转180度
    kernel = np.flip(kernel, axis=(0, 1)).astype('float').ravel()
    # 输出图像
    img_out = np.zeros((oH, oW), dtype='float')
    for i in range(oH):
        for j in range(oW):
            img_out[i, j] = np.dot(img[i:i+h, j:j+w].ravel(), kernel)
    img_out = np.clip(img_out, 0, 255).astype('uint8')
    return img_out

import cv2

img_path = 'Fig3.08(a).jpg'

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 拉普拉斯算子
laplacian = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])
# 图像滤波
img_lap = img_filter(img, laplacian)
# 显示图像
cv2.imwrite('img/origin.jpg', img)
cv2.imwrite('img/laplacian.jpg', img_lap)

