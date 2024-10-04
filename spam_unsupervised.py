import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Sample emails
emails = [
    "Congratulations! You've won a lottery!",
    "Your invoice is attached.",
    "Earn money fast! Click here!",
    "Meeting at 10 AM tomorrow.",
    "Free gift card offer just for you!"
]

#  Convert emails to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(emails)

#  Apply K-Means clustering
num_clusters = 2  
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Analyze clusters
labels = kmeans.labels_
for i, email in enumerate(emails):
    print(f"Email: {email} -> Cluster: {labels[i]}")
