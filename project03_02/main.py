import numpy as np
import cv2
from matplotlib import pyplot as plt

img_path = 'Fig3.08(a).jpg'
sav_path = 'histogram_equalization.png'

# 读入一张灰度图
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
# 显示灰度图
plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 计算直方图，使用numpy的bincount函数
hist = np.bincount(img.ravel(), minlength=256)
# 显示直方图
plt.subplot(222)
plt.bar(range(256), hist)
plt.yscale('log')
plt.xlim([0, 256])
plt.title('Origin Histogram')

# 计算累积分布函数
cdf = hist.cumsum()
cdf = cdf / cdf[-1] * 255 # 归一化到0-255
cdf = cdf.astype(np.uint8) # 转换为整数
# 直方图均衡化
img_eq = cdf[img]
# 显示均衡化后的图像
plt.subplot(223)
plt.imshow(img_eq, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

# 计算均衡化后的直方图
hist_eq = np.bincount(img_eq.ravel(), minlength=256)
# 显示均衡化后的直方图
plt.subplot(224)
plt.bar(range(256), hist_eq)
plt.yscale('log')
plt.xlim([0, 256])
plt.title('Equalized Histogram')

# 显示图像和直方图
plt.suptitle('DIP2E 03-02 Histogram Equalization')
plt.tight_layout()
plt.savefig(sav_path dpi=300)
plt.show()
