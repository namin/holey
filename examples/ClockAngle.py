def sat(hands: List[int], target_angle=45):
      h, m = hands
      assert 0 < h <= 12 and 0 <= m < 60
      hour_angle = 30 * h + m / 2
      minute_angle = 6 * m
      return abs(hour_angle - minute_angle) in [target_angle, 360 - target_angle]
