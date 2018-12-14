# -*- coding: utf-8 -*-
r"""
    Purpose:
        generate encrypted data by python pycrypto library

"""
import random
import string

import numpy as np
from cryptography.fernet import Fernet


def generate_encrypted_data(num=20, size=2):
    """

    :param num:
    :return:
    """
    char_str = string.ascii_lowercase + string.ascii_uppercase + string.digits
    print(char_str)

    X = []
    y = []
    d_X = []
    key = Fernet.generate_key()
    # message = "my deep dark secret".encode()
    f = Fernet(key)
    for i in range(num):
        size_tmp = np.random.randint(size)
        if size_tmp == 0:
            continue
        print(f'size_tmp:{size_tmp}')
        message = [random.choice(char_str) for _ in range(size_tmp)]
        X.append(message)
        message = ''.join(message).encode()
        print(f'plaintxt:{message}')

        # encrypt data
        encrypted = f.encrypt(message)
        print(f'encrypted data:{encrypted}')
        y.append(encrypted.decode())

        # decrypt data
        decrypted = f.decrypt(encrypted)
        print(f'decrypted data: {decrypted}')

    return X, y, d_X


def char_to_int(X):
    """

    :param X:
    :return:
    """
    X_int = []
    max_dim = 0
    for idx, v in enumerate(X):
        v_tmp = []
        for chr in v:
            v_tmp.append(ord(chr))
        if max_dim < len(v):
            max_dim = len(v)
        X_int.append(v_tmp)

    return X_int, max_dim


def align_data(X, max_dim=10):
    """

    :param X:
    :return:
    """
    X_data = []
    for idx, v in enumerate(X):
        if len(v) < max_dim:
            v += [0 for _ in range(max_dim - len(v))]
        X_data.append(v)

    return X_data


if __name__ == '__main__':
    X, y, _ = generate_encrypted_data(num=10)
    X = char_to_int(X)
    y = char_to_int(y)
    print(X)
    print(y)
