class NotEnoughUniqueDatapointsException(BaseException):
    """
    Raised when the number of different datapoints is smaller than the required number of clusters
    """

    def __init__(self, n_unique_dp, k):
        self.ndp = n_unique_dp
        self.k = k

    def __str__(self):
        return f'Not enough unique datapoints (n_unique = {self.ndp}) for required clusters (k = {self.k})'


class KMedoidsNotEnoughLabelsException(BaseException):
    """
    Raised when the number of labels found during k-medoids is lower than n_clusters. This may happen if the distance
    metric is extremely coarse, and the medoids have distance 0 to each other. (low dimensions + high approximations)
    """

    def __init__(self, n_clusters, n_labels):
        self.n_clusters = n_clusters
        self.n_labels = n_labels

    def __str__(self):
        return f'Number of clusters found during k-medoids ({self.n_labels})is less that the number of clusters' \
               f'required ({self.n_clusters}). Consider a less coarse approximation or higher dimensionality.'


