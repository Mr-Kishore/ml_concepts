# K means explained
import random

class KMeans():
    def __init__(self, k=3, iterations=1000):
        self.k = k
        self.iterations = iterations

    def euclidean_distance(self, p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)**0.5
    
    def find_mean(self, cluster):
        return (sum(p[0] for p in cluster) / len(cluster), sum(p[1] for p in cluster) / len(cluster))

    def fit_predict(self, data):
        # clusters = {i+1:[] for i in range(self.k)}
        centroids = random.sample(data, self.k)

        for _ in range(self.iterations):
            for point in data:
                distances = []
                clusters = {i+1:[] for i in range(self.k)}
                for centroid in centroids:
                    distances.append(self.euclidean_distance(point, centroid))
                
                cluster_id = distances.index(min(distances)) + 1
                clusters[cluster_id].append(point)
            
            # New centroids, find mean of all the values
            new_centroids = []
            for cluster_id in clusters.keys():
                new_centroids.append(self.find_mean(clusters[cluster_id]))

            
            if new_centroids == centroids:
                centroids = new_centroids
                break

            centroids = new_centroids

        return centroids, clusters



random_data1 = random.sample(range(0, 1000), 100)

random_data2 = []
for i in range(100):
    random_data2.append((random.uniform(0.0, 1000.0), random.uniform(500.0, 900.0)))

kmeans = KMeans()
centroids, clusters = kmeans.fit_predict(data=random_data2)