import random

class KMeans():
    def __init__(self, k=3, iterations=1000):
        self.k = k
        self.iterations = iterations

    def euclidean_distance(self, p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def find_mean(self, cluster):
        return (sum(p[0] for p in cluster) / len(cluster), sum(p[1] for p in cluster) / len(cluster))

    def fit_predict(self, data):
        centroids = random.sample(data, self.k)

        for _ in range(self.iterations):
            clusters = {i+1: [] for i in range(self.k)}

            for point in data:
                distances = [self.euclidean_distance(point, centroid) for centroid in centroids]
                cluster_id = distances.index(min(distances)) + 1
                clusters[cluster_id].append(point)

            new_centroids = []
            for cluster_id in clusters.keys():
                if clusters[cluster_id]:  # Check if the cluster is not empty
                    new_centroids.append(self.find_mean(clusters[cluster_id]))
                else:

                    new_centroids.append(random.choice(data))

            if new_centroids == centroids:
                break

            centroids = new_centroids

        return centroids, clusters

random_data2 = [(random.uniform(0.0, 1000.0), random.uniform(500.0, 900.0)) for _ in range(100)]

kmeans = KMeans(k=3, iterations=1000)

centroids, clusters = kmeans.fit_predict(data=random_data2)

# Print the results
print("Centroids:", centroids)
print("Clusters:", clusters)
