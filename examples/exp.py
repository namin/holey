def sat(n: int):
    for i in range(5):
        n += n + 1
    return n == 351