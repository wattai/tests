import numpy as np
import functools


def ano(onegai=123):
    a = 13 * onegai
    return np.array([a])


randint = functools.partial(np.random.randint, 0, 10)

if __name__ == "__main__":
    a = randint([3, 3])
    b = randint([3, 1])

    print(a)
    print(b)
    print(a @ b)

