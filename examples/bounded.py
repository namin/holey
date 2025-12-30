def sat(li: List[int]):
    return (len(li) == 5 and
            li.count(li[3]) == 2)