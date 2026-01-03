def sat(s: str):
    return len(s) == 5 and all(c.isupper() or not c.isalpha() for c in s)