import numpy as np
from sklearn.cluster import KMeans

def get_random_centroids(X, k):

    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    
    centroids = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    centroids_indices = np.random.randint(0, X.shape[0], k)
    centroids = X[centroids_indices]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # make sure you return a numpy array
    return np.asarray(centroids).astype(np.float) 



def lp_distance(X, centroids, p=2):

    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    X_repeated = np.repeat(np.expand_dims(X, 0), k, axis=0).astype(float)
    centroids = np.expand_dims(centroids, 1)
    distances = np.sum(np.abs(X_repeated - centroids) ** p, axis=2) ** (1 / p)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return distances

def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    for _iter in range(max_iter):
        # to check later if centroids didn't change
        last_centroids = centroids

        # calculate distances
        rgb_distances_to_each_centroid = lp_distance(X, centroids, p=p)

        # calculate minimum distance centroid, for each RGB value
        classes = np.argmin(rgb_distances_to_each_centroid, axis=0)

        # caculate new centorids, based on minimum distances calculated
        centroids = np.array([np.mean(X[np.where(classes == centroid_index)], axis=0) for centroid_index in range(len(centroids))])

        if np.allclose(last_centroids, centroids):
            # print(f"Stopped after {_iter+1} Iterations, because of no change in centroids values")
            break

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes

def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = None
    centroids = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    centroids = np.array([])
    X_copy = X.copy()

    # 1. choose a centroid, uniformly at random
    random_index = np.random.randint(0, len(X_copy))
    centroids = np.expand_dims(X_copy[random_index], axis=0)  # currently it's just one centroid

    # don't forget to remove it
    np.delete(X_copy, random_index)

    # stages 2 & 3
    for k in range(k):
        # 2. compute distances for each point to each centroid
        rgb_distances_to_each_centroid = lp_distance(X, centroids, p=p)

        # 3. choose a new centroid, based on the prob dist denoted
        min_distance_by_rgb = np.min(rgb_distances_to_each_centroid, axis=0)
        total_squared_distance = np.sum(min_distance_by_rgb ** 2)
        prob_dist = min_distance_by_rgb ** 2 / total_squared_distance

        new_centroid_index = np.random.choice(np.arange(0, X_copy.shape[0]), size=1, p=prob_dist)
        new_centroid = X_copy[new_centroid_index]
        centroids = np.append(centroids, new_centroid, axis=0)

        # don't forget to remove the chosen centroid point, to not get picked again
        np.delete(X_copy, new_centroid_index)


    # 5. Proceed to K-Means, with new 'smart' centroids

    for _iter in range(max_iter):
        # to check later if centroids didn't change
        last_centroids = centroids

        # calculate distances
        rgb_distances_to_each_centroid = lp_distance(X, centroids, p=p)

        # calculate minimum distance centroid, for each RGB value
        classes = np.argmin(rgb_distances_to_each_centroid, axis=0)

        # caculate new centorids, based on minimum distances calculated
        centroids = np.array([np.mean(X[np.where(classes == centroid_index)], axis=0) for centroid_index in range(len(centroids))])

        if np.allclose(last_centroids, centroids):
            # print(f"Stopped after {_iter+1} Iterations, because of no change in centroids values")
            break

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes
