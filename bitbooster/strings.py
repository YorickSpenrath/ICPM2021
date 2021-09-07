# DBSCAN RESULT STATISTICS ---------------------------------------------------------------------------------------------
NUMBER_OF_FOUND_CLUSTERS = 'number_of_found_clusters'
SMALLEST_CLUSTER_SIZE = 'smallest_cluster_size'
LARGEST_CLUSTER_SIZE = 'largest_cluster_size'
MEAN_CLUSTER_SIZE = 'mean_cluster_size'
STD_CLUSTER_SIZE = 'std_cluster_size'
NUMBER_OF_CLUSTERS_AT_LEAST_MIN_PTS = 'number_of_clusters_at_least_min_pts'

# DBSCAN EXECUTION ---------------------------------------------------------------------------------------------------
OUTER_NEIGHBOURHOOD_COMPUTATIONS = f'number_of_outer_neighbourhood_computations'
INNER_NEIGHBOURHOOD_COMPUTATIONS = f'number_of_inner_neighbourhood_computations'
NOISE = 'noise'
NOISE_FRACTION = f'{NOISE}_fraction'
EPS = 'eps'
MIN_POINTS = 'min_points'
UNDEFINED_INT = -2
NOISE_INT = -1

# METRIC----------------------------------------------------------------------------------------------------------------
METRIC_CODE = 'metric_code'
EUC = 'EUC'
MAN = 'MAN'
all_metric_codes = [EUC, MAN]

EUCLIDEAN = 'euclidean'
MANHATTAN = 'manhattan'
WEIGHTED_JACCARD = 'weighted_jaccard'
ALL_METRICS = [EUCLIDEAN, MANHATTAN, WEIGHTED_JACCARD]

# Clustering algorithm -------------------------------------------------------------------------------------------------
CLUSTERING_ALGORITHM = 'clustering_algorithm'
VORONOI = 'voronoi'
DBSCAN = 'dbscan'
all_clustering_algorithms = [VORONOI, DBSCAN]
