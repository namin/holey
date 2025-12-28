def sat(root: float, coeffs=[1, 2, 3, 17]):
    return abs(sum(coeff * (root ** i) for i, coeff in enumerate(coeffs))) < 1e-4