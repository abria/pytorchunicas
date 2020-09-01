# divide the list into parts that differ at most by 1
def partition(list, parts):
    return [list[(i*len(list))//parts:((i+1)*len(list))//parts] for i in range(parts)]