# project 03_03
import numpy as np
# 四则运算函数定义
## 加法 imgA + imgB
img_add = lambda imgA, imgB: np.clip(imgA.astype('int16') + imgB, 0, 255).astype('uint8')
## 减法 imgA - imgB
img_sub = lambda imgA, imgB: np.clip(imgA.astype('int16') - imgB, 0, 255).astype('uint8')
## 乘法 imgA * imgB
img_mul = lambda imgA, imgB: np.clip(np.multiply(imgA.astype('int16'), imgB), 0, 255).astype('uint8')
## 除法，注意此处使用 imgA / imgB * 255
img_div = lambda imgA, imgB: np.clip(np.divide(imgA.astype('float'), imgB), 0, 255).astype('uint8')

import cv2

imgB_path = 'Fig3.15(a)4.jpg'
imgA_path = 'Fig3.15(a)2.jpg'
const = 1.2

imgA = cv2.imread(imgA_path, cv2.IMREAD_GRAYSCALE)
imgB = cv2.imread(imgB_path, cv2.IMREAD_GRAYSCALE)

# 显示原图
cv2.imwrite('img/A.jpg', imgA)
cv2.imwrite('img/B.jpg', imgB)
# 四则运算
cv2.imwrite('img/add.jpg', img_add(imgA, imgB))
cv2.imwrite('img/sub.jpg', img_sub(imgA, imgB))
cv2.imwrite('img/mul.jpg', img_mul(imgA, imgB))
cv2.imwrite('img/div.jpg', img_div(imgA, imgB))
# 常数乘法
cv2.imwrite('img/const_mul.jpg', img_mul(imgA, const))
