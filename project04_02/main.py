import numpy as np
def muled_1xy(arr: np.ndarray) -> np.ndarray:
    '''
    返回 arr 乘 (-1)^(x+y)
    不会改变 arr 的值
    '''
    arr = arr.copy().astype(float)
    arr[::2, 1::2] *= -1
    arr[1::2, ::2] *= -1
    return arr

def mul_func(arr: np.ndarray, func: np.ndarray) -> np.ndarray:
    '''
    返回 arr 乘 func
    要求两者形状相同
    '''
    return np.multiply(arr, func)

def muled_1xy_andreal(arr: np.ndarray) -> np.ndarray:
    '''
    返回 arr 乘 (-1)^(x+y) 并取实部
    不会改变 arr 的值
    '''
    return muled_1xy(arr.real)

def FFT(arr: np.ndarray, inverse: bool = False) -> np.ndarray:
    '''
    自实现的快速傅里叶变换
    要求 arr 的长宽均为 2 的幂次
    inverse 为 True 时表示进行逆变换
    '''
    h, w = arr.shape
    assert (h & (h - 1)) == 0 and (w & (w - 1)) == 0, "FFT: shape must be 2^x * 2^y"

    def fft(arr: np.ndarray, axis: int, inverse: bool = False) -> np.ndarray:
        '''
        一维快速傅里叶变换
        axis 为变换的轴
        '''
        n = arr.shape[axis]
        if n == 1:
            return arr

        # 分治递归
        odd_arr = arr.take(range(0, n, 2), axis)
        even_arr = arr.take(range(1, n, 2), axis)
        odd_fft = fft(odd_arr, axis, inverse)
        even_fft = fft(even_arr, axis, inverse)

        # 计算旋转因子
        inv = 1 if inverse else -1
        Wn = np.exp(inv * 2j * np.pi / n)
        Wn_range = np.power(Wn, range(n // 2))
        # 按axis广播
        Wn_range = np.expand_dims(Wn_range, axis=axis ^ 1)

        # 计算结果
        return np.concatenate([odd_fft + Wn_range * even_fft, odd_fft - Wn_range * even_fft], axis=axis)

    result = arr.copy()
    result = fft(result, 0, inverse)
    result = fft(result, 1, inverse)

    if inverse:
        result /= h * w
    return result

def dft_spectrum(arr: np.ndarray) -> np.ndarray:
    '''
    返回 arr 的频谱
    '''
    return np.abs(FFT(muled_1xy(arr)))

import cv2
from matplotlib import pyplot as plt
img_path = "Fig4.11(a).jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

img_pad = np.pad(img, ((6, 6), (6, 6)))

spectrum = dft_spectrum(img_pad)
# 归一化到 0-255
spectrum = np.log(spectrum + 1)
min_val, max_val = np.min(spectrum), np.max(spectrum)
spectrum = (spectrum - min_val) / (max_val - min_val) * 255
# 保存
cv2.imwrite("spectrum.jpg", spectrum)

plt.subplot(121)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.subplot(122)
plt.imshow(spectrum, cmap="gray")
plt.title("Spectrum Centered: log(1 + |F(u, v)|)")
plt.axis("off")
plt.show()

# 计算图像均值
avg = np.abs(FFT(img_pad)[0, 0]) / img.size
print("average:", avg)
