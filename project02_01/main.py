import numpy as np
import cv2
from matplotlib import pyplot as plt

img_path_format = 'Fig2.22({}).jpg'
sav_path = 'halftoning.png'

class Halftoning:
    def __init__(self, img):
        '''
        img: 2D numpy array, [r, c]
        '''
        # halftone matrix [10, 9]
        self.halftone = np.zeros((10, 9), dtype=np.uint8)
        lst = (1, 8, 0, 6, 2, 5, 7, 3, 4)
        for i in range(1,10):
            self.halftone[i, lst[:i]] = 1

        self.img = img
        self.r, self.c = img.shape

    def __call__(self)->np.ndarray:
        # 量化至 [0,9]
        img_quant = (self.img.astype(np.float32) / 255 * 9).astype(np.uint8)
        # 索引 [r, c] -> [r, c, 10]
        img_index = np.zeros((self.r, self.c, 10), dtype=np.uint8)
        img_index[np.arange(self.r)[:, np.newaxis], np.arange(self.c), img_quant] = 1
        # halftoning [r, c, 10] -> [r, c, 9]
        img_halftone = img_index @ self.halftone
        # [r, c, 9] -> [r, c, 3, 3]
        img_halftone = img_halftone.reshape(self.r, self.c, 3, 3)
        # [r, c, 3, 3] -> [r*3, c*3]
        img_halftone_res = img_halftone.transpose(0, 2, 1, 3).reshape(self.r*3, self.c*3)
        return img_halftone_res



for i, ch in enumerate('abc'):
    # 读入一张灰度图
    img = cv2.imread(img_path_format.format(ch), cv2.IMREAD_GRAYSCALE)
    # 显示灰度图
    plt.subplot(3, 2, 2*i+1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image ({})'.format(ch))
    plt.axis('off')

    halftoning = Halftoning(img)()
    # 显示halftoning后的图像
    plt.subplot(3, 2, 2*i+2)
    plt.imshow(halftoning, cmap='gray')
    plt.title('Halftoning Image ({})'.format(ch))
    plt.axis('off')

# 显示图像和直方图
plt.suptitle('DIP2E 02-01 Halftoning')
plt.tight_layout()
plt.savefig(sav_path, dpi=300)
plt.show()
