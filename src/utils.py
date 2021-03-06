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


def make_index(li):
    zipped = list(zip(li, range(len(li))))

    idx = {z[0]: z[1] for z in zipped}

    return idx


def list2vec(l, idx):
    li = [idx[k] for k in l]
    vec = [len(idx)] * len(idx)

    for i in range(len(li)):
        vec[li[i]] = i

    return vec


def custom_eval(l1, l2, match=lambda x, y: x == y):
    mat = [[0] * (len(l2) + 1)] * (len(l1) + 1)

    for i in range(1, len(l1) + 1):
        for j in range(1, len(l2) + 1):
            if match(l1[i - 1], l2[j - 1]):
                matched = mat[i - 1][j - 1] + 1 / 2 ** max(i, j)
            else:
                matched = 0
            from_up = mat[i - 1][j]
            from_left = mat[i][j - 1]
            mat[i][j] = max(matched, from_up, from_left)

    return mat[len(l1)][len(l2)] / (1 - 1 / 2 ** max(len(l1), len(l2)))


def to_rank(li):
    tmplist = list(zip(sorted(li), range(len(li))))
    tmplist.reverse()
    dd = dict(tmplist)

    return [dd[i]+1 for i in li]


def spearman_rank(l1, l2):
    from scipy.stats import pearsonr
    r1 = to_rank(l1)
    r2 = to_rank(l2)

    return pearsonr(r1,r2)[0]


def kendall_tau(l1, l2):
    r1 = to_rank(l1)
    r2 = to_rank(l2)

    n = len(r1)

    c, nc = (0, 0)

    paired = list(zip(r1,r2))

    for i in range(len(paired)):
        p1 = paired[i]
        for j in range(i+1, len(paired)):
            p2 = paired[j]
            if ((p1[0] - p2[0]) * (p1[1] - p2[1])) > 0:
                c += 1
            elif (p1[0] == p2[0]) or (p1[1] == p2[1]):
                pass
            else:
                nc += 1

    return (c - nc) / (n*(n-1)/2)


def goodman_kruskal_gamma(l1, l2):
    r1 = to_rank(l1)
    r2 = to_rank(l2)

    c, rc = (0, 0)

    paired = list(zip(r1, r2))

    for i in range(len(paired)):
        p1 = paired[i]
        for j in range(i + 1, len(paired)):
            p2 = paired[j]
            if ((p1[0] - p2[0]) * (p1[1] - p2[1])) > 0:
                c += 1
            elif ((p1[0] - p2[0]) * (p1[1] - p2[1])) < 0:
                rc +=1
            else:
                pass

    return (c - rc) / (c + rc)