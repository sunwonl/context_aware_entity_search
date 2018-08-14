def list_split(l, size):
    s = list(range(0, len(l), size))
    e = s[1:] + [len(l)]

    ll = [l[x[0]:x[1]] for x in zip(s, e)]

    return ll


def add_dict(d1, d2):
    from collections import defaultdict

    d3 = defaultdict(list)

    for k in d1:
        for v in d1[k]:
            d3[k].append(v)
    for k in d2:
        for v in d2[k]:
            d3[k].append(v)

    return dict(d3)


def lcs(s1, s2, match=lambda x,y: x==y ):
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if match(s1[x - 1], s2[y - 1]):
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest]


def get_max(iterable, key=lambda x: x):
    max_one = None
    for item in iterable:
        if max_one is None:
            max_one = item
            continue
        elif key(max_one) < key(item):
            max_one = item

    return max_one


def reduce_by_key(func, li, key=lambda x: x, value=lambda x: x):
    from itertools import groupby

    reduced = []
    for k, group in groupby(sorted(li, key=key), key=key):
        reduced.append((k, func(list(map(value, group)))))

    return reduced


