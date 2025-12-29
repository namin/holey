def sat(x: int) -> bool:
  if x > 5:
    if x > 10:
      if x > 20:
        return x % 2 == 0
      else:
        return x % 2 == 1
    else:
      return x < 10
  else:
    return x == 0