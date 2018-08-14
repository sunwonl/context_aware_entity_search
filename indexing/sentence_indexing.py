from elasticsearch.client import IndicesClient as IdxCli
from elasticsearch import Elasticsearch as ES
from utils import *
import time

def index_sentences(articles):
    index_actions = []

    action = {'index': {'_index': 'reuter_sentence', '_type': 'sentence'}}

    stime = time.time()

    all_sents = 0
    indexed_sents = 0

    for a in articles:
        all_sents += len(a['sentences'])
        for s in a['sentences']:
            if len(s[1]) == 0:
                continue

            sentence = {'sentence': s[0],
                        'entity': s[1],
                        'date': a['date'],
                        'section': a['section']
                        }
            index_actions.append(action)
            index_actions.append(sentence)
            indexed_sents += 1

    es.bulk(body=index_actions)
    etime = time.time()

    print('{}/{} : elapsed [{}]s'.format(indexed_sents, all_sents, (etime - stime)))


if __name__ == '__main__':
    import pickle
    try:
        pkl_root = 'D:/data/invest universe/Reuters_EDGAR/crawled/pickled/'
        r = open(pkl_root + 'reuter_corpus_ee.pkl', 'rb')
        corpus_ee = pickle.load(r)
    except Exception as e:
        print(e)

    es = ES('localhost:9200')
    index_client = IdxCli(es)
    sentence_mapping = {
        "dynamic": 'false',
        "properties": {  # field 설명
            "sentence": {"type": "text"},  # 각 field 의 타입 기술
            "date": {"type": "date",
                     'format': 'yyyyMMdd'
                     },
            "section": {"type": "keyword"},
            "entity": {'type': 'text'}
        }
    }

    try:
        index_client.delete(index='reuter_sentence')
    except Exception as e:
        print(e)

    index_client.create(index='reuter_sentence',
                        body={
                            'mappings': {
                                'sentence': sentence_mapping
                            }
                        })

    from multiprocessing.dummy import Pool as ThreadPool

    s = time.time()

    tp = ThreadPool(8)
    article_set = list_split(corpus_ee, 1000)
    tp.map(index_sentences, article_set)
    tp.close()
    tp.join()

    e = time.time()

    print('elapsed time: {}s'.format(e - s))

    a = es.search(q='sentence:"Apple inc"',size=3)

    print(a)
    print('done!')