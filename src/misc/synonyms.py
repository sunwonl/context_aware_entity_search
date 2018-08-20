from search.company_search import *


if __name__ == '__main__':
    data_root = 'D:/data/Invest universe/'

    import pickle

    # some_company = []
    # with open('../../resources/brand_value_interbrand.pkl', 'rb') as r:
    #     df = pickle.load(r)
    #
    # some_company = list(df['Brand'].apply(lambda x:x.lower()))
    #
    # q = 'iphone'
    # res = search_company(q, some_company, verbosity=True, search_size=1000)
    #
    # from pprint import pprint
    #
    # pprint(res)

    from pprint import pprint
    from elasticsearch import Elasticsearch as ES

    q = 'iphone'
    es = ES('localhost:9200')
    result = es.search(q='sentence:({})'.format(q), size=1000)
    result = result['hits']['hits']

    pprint(result)