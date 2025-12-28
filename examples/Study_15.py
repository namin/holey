def sat(li: List[int]):
    return all(sum(li[:i]) == 2 ** i - 1 for i in range(20))