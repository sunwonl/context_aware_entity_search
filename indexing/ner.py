from utils import *


def load_corpus_pkl(reuter_corpus):
    import pickle

    corpus = None
    with open(reuter_corpus, 'rb') as r:
        corpus = pickle.load(r)

    print(len(corpus))

    return corpus


def get_entities(text, nlp):
    from itertools import groupby
    from operator import itemgetter

    non_entity_lables = ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
    doc = nlp(text, disable=['tagger', 'parser'])
    all_ents = [(e.label_, e) for e in doc.ents if e.label_ not in non_entity_lables]

    entities = {}
    for key, group in groupby(sorted(all_ents, key=itemgetter(0)), key=itemgetter(0)):
        entities[key] = [x[1].text for x in group]

    return entities


def extract_entities(doc, fields, ge=get_entities):
    from functools import reduce

    entity_dicts = [ge(doc[f]) for f in fields]

    all_entities = reduce(add_dict, entity_dicts)

    doc.update(all_entities)

    return doc


def ee(nlp, article):
    title = article['title']
    text = article['text']

    title_ = nlp(title, disable=['tagger', 'parser'])
    text_ = nlp(text, disable=['tagger', 'parser'])

    all_orgs = []
    orgs = []
    for e in title_.ents:
        if e.label_ == 'ORG':
            orgs.append(e.text)
    all_orgs.append(orgs)

    sents = [title]
    for s in text_.sents:
        sents.append(s.text)
        orgs = []
        try:
            for e in s.as_doc().ents:
                if e.label_ == 'ORG':
                    orgs.append(e.text)
            all_orgs.append(orgs)
        except:
            all_orgs.append(orgs)
            continue
    article['sentences'] = list(zip(sents, all_orgs))

    return article


def sentence_split_ner(articles):
    from multiprocessing.dummy import Pool as ThreadPool
    import time

    # ee_func = partial(ee, nlp=nlp)

    corpus_ee = []

    for article_list in articles:
        s = time.time()
        tp = ThreadPool(4)

        result = tp.map(ee, article_list)

        tp.close()
        tp.join()

        print(len(result))
        corpus_ee.extend(result)
        e = time.time()

        print('elapsed {} sec'.format(e - s))

    print(len(corpus_ee))

    return corpus_ee


if __name__ == "__main__":
    import pickle

    rootdir = 'D:/data/invest universe/Reuters_EDGAR/crawled/'
    pkl_root = 'D:/data/invest universe/Reuters_EDGAR/crawled/pickled/'
    reuter_corpus = pkl_root + 'reuter_corpus.pkl'

    with open(reuter_corpus, 'rb') as r:
        corpus = pickle.load(r)
    articles = list_split(corpus, 200)
    corpus_ee = sentence_split_ner(articles)

    with open(pkl_root + 'reuter_corpus_ee.pkl', 'wb') as w:
        pickle.dump(corpus_ee, w)