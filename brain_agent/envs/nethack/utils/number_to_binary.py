import numpy as np

def number_to_binary(number, n_bit):
    binary_array = np.zeros([n_bit], dtype=np.uint8)
    str_binary = np.binary_repr(number)
    digits = list(str_binary);
    digits.reverse()  # reverse to start from lower bit

    assert len(digits) <= n_bit, f"number {number} is too big to represent with {n_bit} bits"

    for idx, digit in enumerate(digits):
        binary_array[idx] = np.float32(digit)
    return binary_array