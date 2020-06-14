import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

articles = pd.read_csv('covid19_articles_20200526.csv')

weeks = [('2020-03-15', '2020-03-22'), 
         ('2020-03-22', '2020-03-29'), 
         ('2020-03-29', '2020-04-05'), 
         ('2020-04-05', '2020-04-12'), 
         ('2020-04-12', '2020-04-19')]

for w in weeks:
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(articles[(articles['date']>=w[0]) & (articles['date']<w[1])].content.values)
    
    n = 2
    model = KMeans(n_clusters=n, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)
    
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    
    for i in range(n):
        print(f'Week{w} Cluster {i}:')
        for ind in order_centroids[i, :10]:
            print(f' {terms[ind]}')