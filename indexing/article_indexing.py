from elasticsearch.client import IndicesClient as IdxCli
from elasticsearch import Elasticsearch as ES
from indexing.data_prepration import reuter_articles

if __name__ == '__main__':
    import time

    es = ES('localhost:9200')
    index_client = IdxCli(es)

    article_mapping = {
        "dynamic": 'false',
        "properties": {  # field 설명
            "title": {"type": "text"},  # 각 field 의 타입 기술
            "date": {"type": "date",
                     'format': 'yyyyMMdd'
                     },
            "section": {"type": "keyword"},
            "text": {'type': 'text'}
        }
    }
    try:
        index_client.delete(index='reuter')
    except Exception as e:
        print(e)

    index_client.create(index='reuter',
                        body={
                            'mappings': {
                                'article': article_mapping
                            }
                        })

    mons = ['%02d' % (x) for x in range(1, 13)]
    days = ['%02d' % (x) for x in range(1, 32)]

    for m in mons:
        num_articles = 0
        for d in days:
            try:
                articles = list(map(lambda x: {k: x[k] if k not in ['text'] else ''.join(x[k]) for k in x},
                                    reuter_articles(rootdir, '2017', m, d)))
            except:
                continue

            avg_time = 0
            print('index start 2017.{}.{}'.format(m, d))

            index_actions = []
            s = time.time()
            action = {'index': {'_index': 'reuter', '_type': 'article'}}

            for a in articles:
                index_actions.append(action)
                index_actions.append(a)
            es.bulk(body=index_actions)
            e = time.time()

            print('  elapsed : {} s'.format((e - s) / len(articles)))
        print('done 2017.{}: {} articles'.format(m, num_articles))

        a = es.search(q='text:Apple inc', size=3)

        print(a)
        print('done!')