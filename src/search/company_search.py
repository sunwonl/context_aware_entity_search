from utils import *


def search_entity(q, search_size=200):
    from itertools import product
    from elasticsearch import Elasticsearch as ES
    es = ES('localhost:9200')
    result = es.search(q='sentence:({})'.format(q), size=search_size)
    result = result['hits']['hits']

    companies = []
    for hit in result:
        hits = hit['_source']['entity']
        score = hit['_score']

        companies.extend(product(hits, [score / len(hits)]))
        # companies.extend(hits)
    return companies


def company_name_match(x, y):
    if x == y:
        return True
    elif abs(len(x) - len(y)) == 1:
        if (len(x) > len(y)) and (x[len(x)-1] == '.'):
            return x[:len(x)-1] == y
        elif (len(y) > len(x)) and (y[len(y)-1] == '.'):
            return y[:len(y)-1] == x
        else:
            return False
    return False


def get_companies(KEYWORD, some_company, search_size=200):
    from operator import itemgetter

    search_result = sorted(search_entity(KEYWORD, search_size=search_size), key=itemgetter(1))

    def refine_names(pair):
        name, score = pair
        name = name.lower().strip()
        return (name, score)

    companies = map(refine_names, search_result)

    companies = [x for x in companies if len(x[0]) > 0]

    from itertools import product

    common_word_set = list()

    for x in product(companies, some_company):
        common_words = lcs(x[0][0].split(' '), x[1].split(' '), match=company_name_match)
        if len(common_words) == 0:
            continue
        common_word_set.append((' '.join(common_words), x[0], x[1]))

    return common_word_set


def post_process_compnay_name(common_word_set, verbose=False):
    from collections import defaultdict
    from operator import itemgetter

    with open('../../resources/black_list.txt', 'r') as r:
        not_good_company_name = [l.strip() for l in r.readlines()]

    common_word_set = sorted([c for c in common_word_set if c[0] not in not_good_company_name])

    tmp_dict = defaultdict(list)

    for mapping in common_word_set:
        tmp_dict[mapping[1]].append(mapping)

    selected = []
    for key in tmp_dict:
        candidates = tmp_dict[key]
        selected.append(get_max(candidates, key=itemgetter(0)))
    if verbose:
        return selected
    else:
        return [(x[2], x[1][1]) for x in selected]


def search_company(q, target_company, verbosity=False, search_size=200):
    from operator import itemgetter
    common_word_set = get_companies(q, target_company, search_size=search_size)

    l1 = post_process_compnay_name(common_word_set, verbosity)
    if verbosity:
        res = l1
    else:
        res = reduce_by_key(lambda x: sum(x), l1, key=itemgetter(0), value=itemgetter(1))
        res = sorted(res, key=itemgetter(1), reverse=True)

    return res


if __name__ == '__main__':
    data_root = 'D:/data/Invest universe/'

    import pandas as pd

    sp500_code = pd.read_excel(data_root + 'SP500_3.xlsx')
    some_company = [x.lower() for x in sp500_code.Name]

    sp500_code.head()

    q = 'iphone'
    res = search_company(q, some_company, verbosity=False, search_size=1000)

    from pprint import pprint

    pprint(res)
