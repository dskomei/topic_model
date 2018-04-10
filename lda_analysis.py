from pathlib import Path
import gensim
import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


"""
LDAモデルを使ってトピック分類をする

"""


corpus_dir_path = Path('./corpus')
model_dir_path = Path('./model')
data_dir_path = Path('./data')
result_dir_path = Path('./result')


if not result_dir_path.exists():
    result_dir_path.mkdir(parents=True)


file_name = 'tweet_anime.txt'
with open(data_dir_path.joinpath(file_name.replace('.txt', '_cut.txt')), 'r', encoding='utf-8') as file:
    texts = file.readlines()
texts = [text.replace('\n', '') for text in texts]


dictionary = gensim.corpora.Dictionary.load_from_text(corpus_dir_path.joinpath(file_name.replace('.txt', '_dictionary.dict.txt')).__str__())
corpus = gensim.corpora.MmCorpus(corpus_dir_path.joinpath(file_name.replace('.txt', '_corpus.mm')).__str__())
model = gensim.models.ldamodel.LdaModel.load(model_dir_path.joinpath(file_name.replace('.txt', '_lda.pkl')).__str__())


# 文章ごとのトピック分類結果を得る
topics = [model[c] for c in corpus]

def sort_(x):
    return sorted(x, key=lambda x_:x_[1], reverse=True)

# トピック数を取得
num_topics = model.get_topics().shape[0]

# 文章間の類似度を測定するためにコサイン類似度の行列を作成
# 各文章を行として、トピックの重みを格納
dences = np.zeros((len(topics), num_topics), dtype=np.float)
for row, t_ in enumerate(topics):
    for col, value in t_:
        dences[row, col] = value

# 文章間のコサイン類似度の計算
cosine_ = cosine_similarity(dences, dences)
for i in range(cosine_.shape[0]):
    # 対角成分は類似度が1となるため、０に修正
    cosine_[i, i] = 0

#
# 文章の比較
#
target_doc_id = 4
print(texts[target_doc_id])
print(sort_(topics[target_doc_id]))

print(texts[int(np.argmax(cosine_[target_doc_id]))])
print(sort_(topics[int(np.argmax(cosine_[target_doc_id]))]))

#
# 各トピックの要素の表示
#
topic10 = []
for topic_ in model.show_topics(-1, formatted=False):
    topic10.append([token_[0] for token_ in topic_[1]])

topic10 = pd.DataFrame(topic10)
print(topic10)
topic10.to_csv(result_dir_path.joinpath(file_name.replace('.txt', '_topic10.csv')).__str__(),
               index=False, encoding='utf-8')