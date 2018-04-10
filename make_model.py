from pathlib import Path
import gensim


"""
LDAのモデルを作る

"""


corpus_dir_path = Path('./corpus')
model_dir_path = Path('./model')
if not model_dir_path.exists():
    model_dir_path.mkdir(parents=True)


file_name = 'tweet_anime.txt'
dictionary = gensim.corpora.Dictionary.load_from_text(corpus_dir_path.joinpath(file_name.replace('.txt', '_dictionary.dict.txt')).__str__())
corpus = gensim.corpora.MmCorpus(corpus_dir_path.joinpath(file_name.replace('.txt', '_corpus.mm')).__str__())


# トピック数50のLDAを作る
model = gensim.models.LdaModel(corpus,
                               num_topics=50,
                               id2word=dictionary)

model.save(model_dir_path.joinpath(file_name.replace('.txt', '_lda.pkl')).__str__())


