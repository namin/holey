def sat(m: int, hello=[1, 31, 3, 2, 0, 18, 32, -4, 2, -1000, 3502145, 3502145, 21, 18, 2, 60]):
    return m in hello and not any(m < i for i in hello)