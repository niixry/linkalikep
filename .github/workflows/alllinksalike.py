# тут поиск по всем спарсенным статьям топ 3 схожие ссылки (лимит уменьшен до 10 из вики и 10 из хабра)

import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def parse_habr_articles(topic, limit=10):
    base_url = 'https://habr.com/'
    response = requests.get(base_url)
    if response.status_code != 200:
        print(f"Ошибка загрузки {base_url}")
        return []
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = []

    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('/en/articles/') and href not in articles and not href.endswith('/comments/'):
          articles.append(base_url + href)

        elif href.startswith('/en/companies/') and href not in articles and not href.endswith('/comments/'):
          articles.append(base_url + href)

        if len(articles) >= limit:
            break
    info = []
    for art in articles:
      response = requests.get(art)
      if response.status_code != 200:
          print(f"Ошибка загрузки {art}")
          return None
      soup = BeautifulSoup(response.text, 'html.parser')
      title = soup.find('h1').text.strip() if soup.find('h1') else 'Нет заголовка'
      text = ' '.join([p.text for p in soup.select('p')])
      info.append((title, text, art))
    return info

def parse_wikipedia_articles(topic, limit=10):
    url = f"https://ru.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": topic,
        "srlimit": limit
    }
    response = requests.get(url, params=params)
    results = response.json().get('query', {}).get('search', [])

    articles = []
    for result in results:
        page_url = f"https://ru.wikipedia.org/wiki/{result['title'].replace(' ', '_')}"
        page_response = requests.get(page_url)
        page_soup = BeautifulSoup(page_response.text, 'html.parser')
        text = ' '.join([p.text for p in page_soup.select('p')])
        articles.append((result['title'], text, page_url))

    return articles

topic = "интеллект"
habr_articles = parse_habr_articles(topic)
wiki_articles = parse_wikipedia_articles(topic)

all_articles = habr_articles + wiki_articles
titles, texts, urls = zip(*all_articles)

embeddings = model.encode(texts)
similarity_matrix = cosine_similarity(embeddings)

for i, title in enumerate(titles):
    print(f"\nСтатья: {title} ({urls[i]})")
    similar_indices = np.argsort(similarity_matrix[i])[::-1][1:4]
    similar_indices = [index for index in similar_indices if index < len(titles)]
    for j in similar_indices:
        print(f"  - Похожа на: {titles[j]} ({urls[j]})")

