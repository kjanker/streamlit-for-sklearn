"""
Helper script to provide access to the sklearn models and datasets.
"""
from sklearn.cluster import AffinityPropagation, KMeans, MeanShift, SpectralClustering
from sklearn.datasets import load_iris, load_wine

datasets = {
    "Iris plants dataset": load_iris,
    "Wine recognition dataset": load_wine,
}

models = {
    "KMeans": {
        "class": KMeans,
        "kwargs": {
            "n_clusters": 3,
            "init": ["k-means++", "random"],
            "random_state": 0,
        },
        "kwargs_adv": {
            "n_init": 10,
            "max_iter": 300,
            "tol": 1e-4,
            "algorithm": ["auto", "elkan", "full"],
        },
    },
    "AffinityPropagation": {
        "class": AffinityPropagation,
        "kwargs": {
            "damping": 0.5,
            "random_state": 0,
        },
        "kwargs_adv": {
            "max_iter": 200,
            "convergence_iter": 15,
            "affinity": ["euclidean", "precomputed"],
        },
    },
    "MeanShift": {
        "class": MeanShift,
        "kwargs": {},
        "kwargs_adv": {
            "n_jobs": 1,
            "max_iter": 300,
        },
    },
    "SpectralClustering": {
        "class": SpectralClustering,
        "kwargs": {
            "n_clusters": 3,
            "eigen_solver": ["arpack", "lobpcg", "amg"],
            "random_state": 0,
        },
        "kwargs_adv": {
            "n_components": 8,
            "n_init": 10,
            "gamma": 1.0,
            "affinity": [
                "rbf",
                "nearest_neighbors",
                "precomputed",
                "precomputed_nearest_neighbors",
            ],
            "n_neighbors": 10,
            "eigen_tol": 0.0,
            "assign_labels": ["kmeans", "discretize", "cluster_qr"],
            "degree": 3,
            "coef0": 1,
            "n_jobs": 1,
        },
    },
}
