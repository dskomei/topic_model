from pathlib import Path
import urllib.request
from pyquery import PyQuery as pq

"""
青空文庫から「学問のすすめ」の文章を取得する

"""

data_dir_path = Path('./data')

url = 'https://www.aozora.gr.jp/cards/000296/files/47061_29420.html'
with urllib.request.urlopen(url) as response:
    html = response.read()
query = pq(html,  parser='html')

text = query(".main_text").text().replace('\n', '')
texts = text.split('。')
texts = [text_+'\n' for text_ in texts]

with open(data_dir_path.joinpath('gakumon_no_susume.txt'), 'w', encoding='utf-8') as file:
    file.writelines(texts)

