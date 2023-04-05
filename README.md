# 資料爬蟲、資料庫API串接、文本資料處理

資料爬蟲（Data Crawling）是指利用程式自動抓取網路上的資料，進而建立資料庫或進行資料分析的行為。資料爬蟲通常會使用網頁爬蟲（Web Crawling）的技術來取得網頁上的資訊，也可以利用 API 來獲取資料，或是直接下載公開的資料檔案。以下分別介紹三種類型的資料爬蟲。

1. 檔案來源

檔案來源的資料爬蟲通常是指從公開的資料檔案中，自動下載或提取所需的資料。常見的公開資料檔案包括政府公開資訊、公司財報、股票報價、氣象資訊等。開發者可以使用 Python 程式語言來下載或提取這些資料，並進一步進行分析或儲存。

2. API 來源

API 來源的資料爬蟲通常是指利用 API 伺服器提供的介面，以程式方式獲取資料。API 通常提供結構化的資料，例如 JSON 或 XML 格式的資料，可以使用 Python 中的 requests 模組向 API 伺服器發送 HTTP 請求，獲取回傳的資料後再進行解析或儲存。

API 伺服器通常需要使用者註冊申請一組 API 金鑰（API Key）或 OAuth 認證，以便進行授權，以確保使用者有權限訪問 API 資料。

3. 網頁來源

網頁來源的資料爬蟲通常是指從網頁上自動抓取需要的資料，進而建立資料庫或進行資料分析。這類資料爬蟲通常需要使用網頁爬蟲技術，例如使用 Python 中的 beautifulsoup 套件或 scrapy 框架等，解析 HTML 或 XML 結構化資料，抓取所需的資料。需要注意的是，在進行網頁爬蟲時，必須遵守網站的使用條款和版權法律，避免侵犯他人權益。

利用 Python 收集來自 API 的資料
API 來源的爬蟲主要是通過網路上提供的 API 接口，從網路上獲取數據，例如從某個網站獲取股票市場行情、氣象預報等，通常是先向 API 接口發送請求，獲取 JSON 格式的數據，然後使用 json 模組對數據進行解析和處理。在今天的例子中，我們將以 Python 實現 API 來源的資料收集。若要進行 API 資料收集，可以使用 Python 中的 requests 套件向 API 伺服器發送 HTTP 請求，獲取 API 回傳的資料。以下是一個簡單的範例：

import requests
import json


# 利用 requests 對 API 來源發送一個請求
url = '{ API 網址}'
response = requests.get(url)

# 將請求回應的內容存成一個字串格式
d = response.text

# 將長得像 json 格式的字串解析成字典或列表
data = json.loads(d)

print(data)
以上範例中的 `data` 變數，就是我們利用 Python 抓取回來的資料並且經過解析後變成一個 Python 內建的資料結構。但是有這樣的資料之後，下一步還需要做什麼呢？這個時候就會需要仰賴你對程式語言的熟悉程度，將原始的資料整理成你想要使用的樣子。

資料爬蟲的未來待續
資料爬蟲的應用範圍非常廣泛，幾乎可以涵蓋所有需要自動獲取網路上資料的場景。以下是一些常見的資料爬蟲應用場景：

電子商務產品價格比較
新聞媒體文章抓取
社交網站數據收集和分析
股票報價和交易資料收集和分析
網路搜索引擎資料收集和分析
運動比賽數據抓取和分析
酒店和旅遊預訂資料抓取和比較
除此之外，你還有利用更多的第三方套件與函式庫協助你在資料收集的項目中完成更高階的操作：

requests：用於向網站發送 HTTP 請求，獲取網站上的資訊。
BeautifulSoup：用於解析 HTML 或 XML 結構化資料，抓取所需的資料。
Scrapy：用於編寫網頁爬蟲框架，支持非同步網路爬蟲。
Selenium：用於自動化瀏覽器操作，支持 JavaScript 用於模擬操作網頁的行為解決網頁動態加載的問題。
pandas：用於數據處理和分析，可將爬蟲收集到的數據進行結構化處理和分析。

什麼是資料庫？
資料庫是一種用於儲存和管理大量資料的系統，可以方便地進行資料查詢、更新和刪除等操作。資料庫可以分為關聯式和非關聯式兩種，其中關聯式資料庫使用結構化查詢語言（SQL）來管理資料。SQL 是一種用於管理關聯式資料庫的程式語言，它可以對資料庫進行查詢、插入、更新、刪除等操作。SQL 的語法非常直觀，易於學習和使用。

在 Python 中，我們可以使用 SQLite 庫來創建和操作關聯式資料庫。SQLite 是一種輕量級的關聯式資料庫，它不需要單獨的伺服器進程或配置，可以直接使用 Python 中的內置函數來創建和操作 SQLite 資料庫。使用 Python 與 SQLite，我們可以輕鬆地進行資料庫操作，例如創建和刪除資料表、插入和更新資料、查詢和排序資料等等。這些操作可以幫助我們有效地管理大量資料，並且可以在 Python 程式中方便地使用這些資料。


利用 Python 操作 SQLite 資料庫
SQLite 是一個輕量級的關聯式資料庫管理系統，並且支援 SQL 語言。Python 提供了 SQLite3 模組，讓使用者可以透過 Python 程式碼來操作 SQLite 資料庫。

建立 SQLite 資料庫

在 Python 程式碼中，我們可以使用 SQLite3 模組中的 connect() 函數來建立 SQLite 資料庫：

import sqlite3

conn = sqlite3.connect('example.db')
以上的程式碼會建立一個名為 "example.db" 的 SQLite 資料庫。如果該資料庫不存在，則會自動建立。

在資料庫中建立資料表

在 SQLite 資料庫中，我們需要建立資料表來存放資料。我們可以使用 execute() 方法來執行 SQL 語句來建立資料表：

import sqlite3

conn = sqlite3.connect('example.db')
c = conn.cursor()

c.execute('''CREATE TABLE stocks
             (date text, trans text, symbol text, qty real, price real)''')

conn.commit()
conn.close()
以上的程式碼會建立一個名為 "stocks" 的資料表，並且該資料表包含五個欄位：日期、交易類型、股票代號、數量、價格。這個 SQL 語句會在 SQLite 資料庫中創建一個名為 stocks 的資料表，以下是這個 stocks 資料表的欄位和屬性：


這個資料表可以儲存股票交易的資料，每一筆資料包含了交易日期（date）、交易類型（trans）、股票代號（symbol）、交易數量（qty）和交易價格（price）等欄位。

在資料表中新增一筆資料

在 SQLite 資料庫中，我們可以使用 execute() 方法來執行 SQL 語句來新增資料到資料表中：

import sqlite3

conn = sqlite3.connect('example.db')
c = conn.cursor()

c.execute("INSERT INTO stocks VALUES ('2022-03-01', 'BUY', 'AAPL', 100, 135.0)")

conn.commit()
conn.close()
以上的程式碼會新增一筆資料到 "stocks" 資料表中，該筆資料的日期為 2022-03-01，交易類型為 BUY，股票代號為 AAPL，數量為 100，價格為 135.0。這個 SQL 語句會向名為 stocks 的資料表中插入一筆新資料，以下是插入資料後的樣子：


在資料表中修改資料

在 SQLite 資料庫中，我們可以使用 execute() 方法來執行 SQL 語句來修改資料表中的資料：

import sqlite3

conn = sqlite3.connect('example.db')
c = conn.cursor()

c.execute("UPDATE stocks SET qty = 200 WHERE symbol = 'AAPL'")

conn.commit()
conn.close()
以上的程式碼會修改 "stocks" 資料表中符合股票代號為 AAPL 的資料，將該筆資料的數量修改為 200。

從資料表中刪除資料

在 SQLite 資料庫中，我們可以使用 execute() 方法來執行 SQL 語句來刪除資料表中的資料：

import sqlite3

conn = sqlite3.connect('example.db')
c = conn.cursor()

c.execute("DELETE FROM stocks WHERE symbol = 'AAPL'")

conn.commit()
conn.close()
以上的程式碼會刪除 "stocks" 資料表中符合股票代號為 AAPL 的資料。

從資料表中查詢資料

在 SQLite 資料庫中，我們可以使用 execute() 方法來執行 SQL 語句來查詢資料表中的資料。

使用 fetchone() 方法來取得查詢結果的第一筆資料
使用 fetchall() 方法來取得所有查詢結果的資料
import sqlite3

conn = sqlite3.connect('example.db')
c = conn.cursor()

c.execute("SELECT * FROM stocks WHERE symbol = 'AAPL'")
print(c.fetchone())

conn.close()
這段程式碼會查詢 "stocks" 資料表中符合股票代號為 AAPL 的資料，並且印出查詢結果的第一筆資料。

以上就是在 Python 中使用 SQLite 操作資料庫的基本步驟。學會這些基本的操作之後，可以進一步了解更多進階的 SQ 操作方法，例如 JOIN 語句、索引等等，來更有效地管理和操作資料庫。

文本處理
資料庫Python 提供了許多用於處理文本的庫和模組。以下是一些常用的操作與工具：

正則表達式

正則表達式是一種強大的文本處理工具，可用於搜索、替換和提取文本。Python 內置了 re 函式庫，用於處理正則表達式。例如：

import re

text = 'The quick brown fox jumps over the lazy dog.'
result = re.search('fox', text)
print(result.group(0)) # fox
NLTK

Natural Language Toolkit (NLTK) 是一個 Python 套件，提供了許多自然語言處理的工具和資源。它可以用於分詞、標記、詞性標注和文本分析。例如：

import nltk

text = 'The quick brown fox jumps over the lazy dog.'
tokens = nltk.word_tokenize(text)
print(tokens) # ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']
Jieba

對於中文的自然語言處理，Jieba 是一個流行的 Python 函式庫。Jieba 提供了分詞、詞性標注、關鍵詞提取等功能，是進行中文文本處理的好幫手。以下是一些 Jieba 的使用範例：

(1) 將中文文本分成詞語，可以使用 Jieba 的 cut 方法。例如：

import jieba

text = '我喜歡用Python編程'
words = jieba.cut(text)
print('/'.join(words)) # 我喜/歡用/Python/編程
(2) 對中文文本進行詞性標注，可以使用 Jieba 的 posseg 方法。例如：

import jieba.posseg as pseg

text = '我喜歡用Python編程'
words = pseg.cut(text)
for word, pos in words:
    print(word, pos)
(3) 提取中文文本的關鍵詞，可以使用 Jieba 的 extract_tags 方法。例如：

import jieba.analyse

text = '我喜歡用Python編程'
keywords = jieba.analyse.extract_tags(text, topK=3)
print(keywords) # ['Python', '喜歡', '編程']
自然語言處理
在自然語言處理中，您可以使用 Python 來進行各種文本分析、語言模型、情感分析等工作。以下是一些常用的自然語言處理庫：

TextBlob

TextBlob 是一個基於 NLTK 的 Python 库，提供了一個簡單的 API，用於文本分析、情感分析和主題分析。例如：

from textblob import TextBlob

text = 'I love Python!'
blob = TextBlob(text)
print(blob.sentiment.polarity) # 0.5
gensim

gensim 是一個用於主題建模和相似度比較的 Python 函式庫。它可以用於建立文檔、詞彙和主題模型，以及進行文本相似度比較。例如：

from gensim import corpora, models, similarities

documents = [
    "The quick brown fox jumps over the lazy dog",
    "A king's breakfast has sausages, ham, bacon, eggs, toast and beans", 
    "I love Python and machine learning"
]
texts = [[word for word in document.lower().split()] for document in documents]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary))
query = "Python and machine learning"
query_bow = dictionary.doc2bow(query.lower().split())
sims = index[tfidf[query_bow]]
print(list(enumerate(sims)))
# [(0, 0.0), (1, 0.024158917), (2, 0.78141415)]
