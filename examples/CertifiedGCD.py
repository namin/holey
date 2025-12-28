def sat(ans: List[int], m=200004931, n=66679984):
    gcd, a, b = ans
    return m % gcd == n % gcd == 0 and a * m + b * n == gcd and gcd > 0