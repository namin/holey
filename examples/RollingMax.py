def sat(maxes: List[int], nums=[-15, -6]):
    for i in range(len(nums)):
        if i > 0:
             assert maxes[i] == max(maxes[i - 1], nums[i])
        else:
            assert maxes[0] == nums[0]
    return True