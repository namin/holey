def sat(depths: List[int], parens="()()(())()(())"):
    groups = parens.split()
    for depth, group in zip(depths, groups):
        budget = depth
        success = False
        for c in group:
            if c == '(':
                budget -= 1
                if budget == 0:
                    success = True
                assert budget >= 0
            else:
                assert c == ')'
                budget += 1
        assert success

    return len(groups) == len(depths)