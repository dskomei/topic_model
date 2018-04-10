from pathlib import Path
from janome.charfilter import *
from janome.analyzer import Analyzer
from janome.tokenizer import Tokenizer
from janome.tokenfilter import *
from gensim import corpora
import re

"""
アニムを含むtweetのコーパスを作る

"""


data_dir_path = Path('./data')
corpus_dir_path = Path('./corpus')
if not corpus_dir_path.exists():
    corpus_dir_path.mkdir(parents=True)


file_name = 'tweet_anime.txt'
with open(data_dir_path.joinpath(file_name), 'r', encoding='utf-8') as file:
    texts = file.readlines()


texts = [text.replace('\n', '') for text in texts]



# janomeのAnalyzerを使うことで、文の分割と単語の正規化をまとめて行うことができる
# 文に対する処理のまとめ
char_filters = [UnicodeNormalizeCharFilter(),                               # UnicodeをNFKC(デフォルト)で正規化
                RegexReplaceCharFilter('http[a-z!-/:-@[-`{-~]', ''),        # urlの削除
                RegexReplaceCharFilter('@[a-zA-Z]+', ''),                   # @ユーザ名の削除
                RegexReplaceCharFilter('[!-/:-@[-`{-~♪♫♣♂✨дд∴∀♡☺➡〃∩∧⊂⌒゚≪≫•°。、♥❤◝◜◉◉★☆✊≡ø彡「」『』○≦∇✿╹◡✌]', ''), # 記号の削除
                RegexReplaceCharFilter('\u200b', ''),                       # 空白の削除
                RegexReplaceCharFilter('アニメ', '')                         # 検索キーワードの削除
                ]

tokenizer = Tokenizer()


#
# 名詞中の数(漢数字を含む)を全て0に置き換えるTokenFilterの実装
#
class NumericReplaceFilter(TokenFilter):
    """
    名詞中の数(漢数字を含む)を全て0に置き換えるTokenFilterの実装
    """
    def apply(self, tokens):
        for token in tokens:
            parts = token.part_of_speech.split(',')
            if (parts[0] == '名詞' and parts[1] == '数'):
                token.surface = '0'
                token.base_form = '0'
                token.reading = 'ゼロ'
                token.phonetic = 'ゼロ'

            yield token

#
#  ひらがな・カタガナ・英数字の一文字しか無い単語は削除
#
class OneCharacterReplaceFilter(TokenFilter):

    def apply(self, tokens):
        for token in tokens:

            # 上記のルールの一文字制限で引っかかった場合、その単語を無視
            if re.match('^[あ-んア-ンa-zA-Z0-9ー]$', token.surface):
                continue

            if re.match('^w+$', token.surface):
                continue

            if re.match('^ー+$', token.surface):
                continue

            yield token


token_filters = [NumericReplaceFilter(),       # 名詞中の漢数字を含む数字を0に置換
                 CompoundNounFilter(),         # 名詞が連続する場合は複合名詞にする
                 POSKeepFilter(['名詞']),      # 名詞・動詞・形容詞・副詞のみを取得する
                 LowerCaseFilter(),            # 英字は小文字にする
                 OneCharacterReplaceFilter(),
                 ]


analyzer = Analyzer(char_filters, tokenizer, token_filters)


tokens_list = []
raw_texts = []
for text in texts:
    text_ = [token.base_form for token in analyzer.analyze(text)]
    if len(text_) > 0:
        tokens_list.append([token.base_form for token in analyzer.analyze(text)])
        raw_texts.append(text)


# 正規化された際に一文字もない文の削除後の元テキストデータ
raw_texts = [text_+'\n' for text_ in raw_texts]
with open(data_dir_path.joinpath(file_name.replace('.txt', '_cut.txt')), 'w', encoding='utf-8') as file:
    file.writelines(raw_texts)

# 単語リストの作成
words = []
for text in tokens_list:
    words.extend([word+'\n' for word in text if word != ''])
with open(corpus_dir_path.joinpath(file_name.replace('.txt', '_word_list.txt')), 'w', encoding='utf-8') as file:
    file.writelines(words)


#
# 単語のインデックス化
#
dictionary = corpora.Dictionary(tokens_list)
# 単語の出現回数が１以下か、文を通しての出現割合が６割を超えているものは削除
dictionary.filter_extremes(no_below=1, no_above=0.6)
dictionary.save_as_text(corpus_dir_path.joinpath(file_name.replace('.txt', '_dictionary.dict.txt')).__str__())

#
# 文の数値ベクトル化
#
corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]
corpora.MmCorpus.serialize(corpus_dir_path.joinpath(file_name.replace('.txt','_corpus.mm')).__str__(), corpus)


