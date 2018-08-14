import json


def reuter_articles(rootdir, y, m, d):
    fname = '{root}/{year}/{mon}/{day}.json'.format(root=rootdir, year=y, mon=m, day=d)

    with open(fname, 'r') as r:
        for l in r:
            if len(l.strip()) == 0:
                continue
            x = json.loads(l)
            article = {k: x[k] if k not in ['text'] else ''.join(x[k]) for k in x}
            yield article


def make_corpus_to_pkl(rootdir, mons, days):
    import pickle

    all_articles = []

    for m in mons:
        for d in days:
            try:
                articles = reuter_articles(rootdir, '2017', m, d)
                all_articles.extend(articles)
            except Exception as e:
                print(e)
                continue

    with open(pkl_root + 'reuter_corpus.pkl', 'wb') as w:
        pickle.dump(all_articles, w)


if __name__ == '__main__':
    rootdir = 'D:/data/invest universe/Reuters_EDGAR/crawled/'
    pkl_root = 'D:/data/invest universe/Reuters_EDGAR/crawled/pickled/'
    reuter_corpus = pkl_root + 'reuter_corpus.pkl'

    year = '2017'
    mon = '01'
    day = '01'
    mons = ['%02d' % (x) for x in range(1, 13)]
    days = ['%02d' % (x) for x in range(1, 32)]

    make_corpus_to_pkl(rootdir=rootdir, mons=mons, days=days)