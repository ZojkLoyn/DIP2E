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

# ==================== 测试 ====================

def test_muled_1xy():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("=== test_muled_1xy ===")
    print("muled_1xy:", muled_1xy(a))
    print("origin:", a)

def test_muled_1xy_andreal():
    a = np.array([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j], [7+7j, 8+8j, 9+9j]])
    print("=== test_muled_1xy_andreal ===")
    print("muled_1xy_andreal:", muled_1xy_andreal(a))
    print("origin:", a)

def test_FFT():
    a = np.zeros((4, 4), dtype=float)
    a[2, 0] -= 7.5
    a[1, 3] += 3.5
    a[1::2, 1::2] += 1
    a[::3, 1:] += 1
    a[1:, ::2] += 1
    a[1:, :-2] += 1
    a[::3, ::2] += 1
    print("=== test_FFT ===")
    print("FFT:", FFT(a))
    print("IFFT-FFT:", FFT(FFT(a), True).real)
    print("origin:", a)

def test_func():
    a = np.array([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j], [7+7j, 8+8j, 9+9j]])
    b = np.array([[1, 0, 1], [0, 2, 0], [3, 0, 3]])
    print("=== test_func ===")
    print("mul_func:", mul_func(a, b))
    print("origin:", a)

test_muled_1xy()
test_muled_1xy_andreal()
test_FFT()
test_func()
