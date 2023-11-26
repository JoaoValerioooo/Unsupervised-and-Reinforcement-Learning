import numpy as np
import math
from sklearn.cluster import KMeans
import skfuzzy as fuzz
from tabulate import tabulate

class MODELS():

    def __init__(self, f):
        self.f = f

    #################################################################################
    #################################################################################
    #################################################################################
    #                       GENERAL METHOD: RUN ALL THE MODELS                      #
    #################################################################################
    #################################################################################
    #################################################################################

    def run_all_models_on_Reuters(self, reuters_1_lsa, reuters_2_lsa, reuters_3_lsa, reuters_1_tags, reuters_2_tags, reuters_3_tags, k=10, number_of_runs=50, treshold = [0.1003, 0.15, 0.1003]):
        """
        Runs all clustering models on the Reuters dataset and calculates precision, recall, and F1-measure for each model.

        Args:
        - reuters_1_lsa: the LSA reduced dataset for Reuters-1
        - reuters_2_lsa: the LSA reduced dataset for Reuters-2
        - reuters_3_lsa: the LSA reduced dataset for Reuters-3
        - reuters_1_tags: the tags for Reuters-1
        - reuters_2_tags: the tags for Reuters-2
        - reuters_3_tags: the tags for Reuters-3
        - k: the number of clusters to form (default=10)
        - number_of_runs: the number of times to run the models (default=50)
        - treshold: the convergence threshold for the Fuzzy K-Means algorithm (default=[0.1003, 0.15, 0.1003])
        """
        precision_K_Means, recall_K_Means, F1_Measure_K_Means = 0.0, 0.0, 0.0
        precision_Fuzzy_K_Means, recall_Fuzzy_K_Means, F1_Measure_Fuzzy_K_Means = 0.0, 0.0, 0.0
        precision_okm, recall_okm, F1_Measure_okm = 0.0, 0.0, 0.0
        columns = ["Algorithm", "Reuters-1", "Reuters-2", "Reuters-3"]
        rows = [["KM", 0, 0, 0],
                ["FKM", 0, 0, 0],
                ["OKM", 0, 0, 0]]
        for X, reuters_tags, num, th in zip((reuters_1_lsa, reuters_2_lsa, reuters_3_lsa),
                                            (reuters_1_tags, reuters_2_tags, reuters_3_tags),
                                            [1, 2, 3], treshold):
            for run in range(0, number_of_runs):
                # Get the same set of prototypes for the models
                prototypes = self.PROTOTYPE_INITIALIZER(X, k)
                # K-Means Crisp Model
                p, r, f = self.K_Means(X, reuters_tags, k, prototypes)
                precision_K_Means = precision_K_Means + p
                recall_K_Means = recall_K_Means + r
                F1_Measure_K_Means = F1_Measure_K_Means + f
                # Fuzzy-K-Means Model
                p, r, f = self.Fuzzy_K_Means(X, reuters_tags, k, prototypes, th)
                precision_Fuzzy_K_Means = precision_Fuzzy_K_Means + p
                recall_Fuzzy_K_Means = recall_Fuzzy_K_Means + r
                F1_Measure_Fuzzy_K_Means = F1_Measure_Fuzzy_K_Means + f
                # OKM Model
                p, r, f = self.okm(X, reuters_tags, k, 100, 1e-6, prototypes)
                precision_okm = precision_okm + p
                recall_okm = recall_okm + r
                F1_Measure_okm = F1_Measure_okm + f

            precision_K_Means, recall_K_Means, F1_Measure_K_Means = precision_K_Means/number_of_runs,\
                                                                    recall_K_Means/number_of_runs,\
                                                                    F1_Measure_K_Means/number_of_runs

            precision_Fuzzy_K_Means, recall_Fuzzy_K_Means, F1_Measure_Fuzzy_K_Means = precision_Fuzzy_K_Means/number_of_runs,\
                                                                                      recall_Fuzzy_K_Means/number_of_runs,\
                                                                                      F1_Measure_Fuzzy_K_Means/number_of_runs

            precision_okm, recall_okm, F1_Measure_okm = precision_okm/number_of_runs,\
                                                        recall_okm/number_of_runs,\
                                                        F1_Measure_okm/number_of_runs

            rows[0][num] = F1_Measure_K_Means
            rows[1][num] = F1_Measure_Fuzzy_K_Means
            rows[2][num] = F1_Measure_okm

            precision_K_Means, recall_K_Means, F1_Measure_K_Means = 0.0, 0.0, 0.0
            precision_Fuzzy_K_Means, recall_Fuzzy_K_Means, F1_Measure_Fuzzy_K_Means = 0.0, 0.0, 0.0
            precision_okm, recall_okm, F1_Measure_okm = 0.0, 0.0, 0.0

        print("Reuters Results:\n\n")
        self.f.write("Reuters Results:\n\n")
        print(tabulate(rows, headers=columns))
        self.f.write(tabulate(rows, headers=columns))

    def run_all_models_on_Genetic(self, X, labels, k=10, number_of_runs=50, treshold = 0.07144):
        """
        Runs K-Means, Fuzzy-K-Means, and OKM models on the given data and calculates precision, recall, and F1-measure for each model.

        Args:
        - X: data array of shape (n_samples, n_features)
        - labels: true labels array of shape (n_samples,)
        - k: number of clusters
        - number_of_runs: number of runs to perform
        - treshold: fuzzy-K-Means treshold
        """
        precision_K_Means, recall_K_Means, F1_Measure_K_Means = 0.0, 0.0, 0.0
        precision_Fuzzy_K_Means, recall_Fuzzy_K_Means, F1_Measure_Fuzzy_K_Means = 0.0, 0.0, 0.0
        precision_okm, recall_okm, F1_Measure_okm = 0.0, 0.0, 0.0
        columns = ["Algorithm", "Precision", "Recall", "F-Measure"]
        rows = [["KM", 0, 0, 0],
                ["FKM", 0, 0, 0],
                ["OKM", 0, 0, 0]]
        for run in range(0, number_of_runs):
            # Get the same set of prototypes for the models
            prototypes = self.PROTOTYPE_INITIALIZER(X, k)
            # K-Means Crisp Model
            p, r, f = self.K_Means(X, labels, k, prototypes)
            precision_K_Means = precision_K_Means + p
            recall_K_Means = recall_K_Means + r
            F1_Measure_K_Means = F1_Measure_K_Means + f
            # Fuzzy-K-Means Model
            p, r, f = self.Fuzzy_K_Means(X, labels, k, prototypes, treshold)
            precision_Fuzzy_K_Means = precision_Fuzzy_K_Means + p
            recall_Fuzzy_K_Means = recall_Fuzzy_K_Means + r
            F1_Measure_Fuzzy_K_Means = F1_Measure_Fuzzy_K_Means + f
            # OKM Model
            p, r, f = self.okm(X, labels, k, 100, 1e-6, prototypes)
            precision_okm = precision_okm + p
            recall_okm = recall_okm + r
            F1_Measure_okm = F1_Measure_okm + f

        precision_K_Means, recall_K_Means, F1_Measure_K_Means = precision_K_Means / number_of_runs, \
                                                                recall_K_Means / number_of_runs, \
                                                                F1_Measure_K_Means / number_of_runs

        precision_Fuzzy_K_Means, recall_Fuzzy_K_Means, F1_Measure_Fuzzy_K_Means = precision_Fuzzy_K_Means / number_of_runs, \
                                                                                  recall_Fuzzy_K_Means / number_of_runs, \
                                                                                  F1_Measure_Fuzzy_K_Means / number_of_runs

        precision_okm, recall_okm, F1_Measure_okm = precision_okm / number_of_runs, \
                                                    recall_okm / number_of_runs, \
                                                    F1_Measure_okm / number_of_runs

        rows[0][1] = precision_K_Means
        rows[0][2] = recall_K_Means
        rows[0][3] = F1_Measure_K_Means

        rows[1][1] = precision_Fuzzy_K_Means
        rows[1][2] = recall_Fuzzy_K_Means
        rows[1][3] = F1_Measure_Fuzzy_K_Means

        rows[2][1] = precision_okm
        rows[2][2] = recall_okm
        rows[2][3] = F1_Measure_okm

        print("Genetic Results:\n\n")
        self.f.write("\n\n\nGenetic Results:\n\n")
        print(tabulate(rows, headers=columns))
        self.f.write(tabulate(rows, headers=columns))

    #################################################################################
    #################################################################################
    #################################################################################
    #           GENERAL METHODS: PROTOTYPE INITIALIZER & F1-MEASUREMENT             #
    #################################################################################
    #################################################################################
    #################################################################################

    def PROTOTYPE_INITIALIZER(self, X, k=10):
        """
        Initialize the prototypes randomly from the dataset

        Parameters:
        X: numpy.ndarray, data set
        k: int, number of clusters, default=10

        Returns:
        numpy.ndarray, the prototypes for each cluster
        """
        return X[np.random.choice(X.shape[0], k, replace=False)]

    def f1_measure(self, coverage, reuters_tags):
        """
        Calculate the F1 measure based on the cluster coverage and the Reuters tags

        Parameters:
        coverage: numpy.ndarray, boolean coverage matrix for each cluster and instance
        reuters_tags: list, the Reuters tags for each instance in the dataset

        Returns:
        tuple of float: the precision, recall, and F1 measure scores
        """
        number_of_identified_linked_pairs = 0
        number_of_correctly_identified_linked_pairs = 0
        number_of_true_linked_pairs = 0

        for instance_id_1 in range(0, len(coverage)):
            for instance_id_2 in range(instance_id_1, len(coverage)):

                for rt in reuters_tags[instance_id_1]:
                    if rt in reuters_tags[instance_id_2]:
                        number_of_true_linked_pairs = number_of_true_linked_pairs + 1
                        break

                for each_cluster_result in range(len(coverage[instance_id_1])):
                    if coverage[instance_id_1][each_cluster_result] == True and coverage[instance_id_2][
                        each_cluster_result] == True:
                        number_of_identified_linked_pairs = number_of_identified_linked_pairs + 1

                        for rt in reuters_tags[instance_id_1]:
                            if rt in reuters_tags[instance_id_2]:
                                number_of_correctly_identified_linked_pairs = number_of_correctly_identified_linked_pairs + 1
                                break
                        break

        precision = number_of_correctly_identified_linked_pairs / number_of_identified_linked_pairs
        recall = number_of_correctly_identified_linked_pairs / number_of_true_linked_pairs
        F1_Measure = (2 * precision * recall) / (precision + recall)
        return precision, recall, F1_Measure

    #################################################################################
    #################################################################################
    #################################################################################
    #                           K-MEANS - Crisp Partitions                          #
    #################################################################################
    #################################################################################
    #################################################################################

    def K_Means_(self, X, k, initializer):
        """
        Apply K-Means clustering on the input data.

        Parameters:
        X (numpy.ndarray): Input data of shape (n_samples, n_features).
        k (int): Number of clusters.
        initializer: The method used to initialize the cluster centers.

        Returns:
        labels (numpy.ndarray): The cluster labels assigned to each data point in X.
        """
        # Create a KMeans object with k clusters
        kmeans = KMeans(n_clusters=k, init=initializer)
        # Fit the data to the KMeans model
        kmeans.fit(X)
        # Get the labels assigned to each data point
        labels = kmeans.labels_
        return labels

    def get_K_Means_coverage(self, labels, X, k):
        """
        Compute the coverage matrix for K-Means clustering.

        Parameters:
        labels (numpy.ndarray): The cluster labels assigned to each data point in X.
        X (numpy.ndarray): Input data of shape (n_samples, n_features).
        k (int): Number of clusters.

        Returns:
        coverage (numpy.ndarray): Coverage matrix of shape (n_samples, k).
        """
        coverage = np.zeros((X.shape[0], k), dtype=bool)
        for index in range(len(labels)):
            coverage[index][labels[index]] = True
        return coverage

    def K_Means(self, X, reuters_tags, k=10, initializer=None):
        """
        Apply K-Means clustering on the input data and evaluate the performance.

        Parameters:
        X (numpy.ndarray): Input data of shape (n_samples, n_features).
        reuters_tags (numpy.ndarray): The ground truth labels for the input data.
        k (int): Number of clusters.
        initializer: The method used to initialize the cluster centers.

        Returns:
        precision (float): Precision score of the K-Means clustering.
        recall (float): Recall score of the K-Means clustering.
        F1_Measure (float): F1-measure score of the K-Means clustering.
        """
        labels = self.K_Means_(X, k, initializer)
        coverage = self.get_K_Means_coverage(labels, X, k)
        precision, recall, F1_Measure = self.f1_measure(coverage, reuters_tags)
        return precision, recall, F1_Measure

    #################################################################################
    #################################################################################
    #################################################################################
    #                                Fuzzy-K-MEANS                                  #
    #################################################################################
    #################################################################################
    #################################################################################

    def Fuzzy_K_Means_(self, X, k, initializer):
        """
        Perform Fuzzy K-Means clustering on the input data.

        Args:
        - X: Input data to cluster.
        - k: Number of clusters to form.
        - initializer: Method for initialization.

        Returns:
        - u.T: Membership matrix indicating the degree of association of each data point to each cluster.
        """
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X.T, k, 2, error=0.005, maxiter=1000, init=None)
        return u.T

    def get_Fuzzy_K_Means_coverage(self, fuzzy_matrix, X, k, threshold):
        """
        Calculate coverage of data points in each cluster based on a threshold value.

        Args:
        - fuzzy_matrix: Membership matrix indicating the degree of association of each data point to each cluster.
        - X: Input data used for clustering.
        - k: Number of clusters.
        - threshold: Threshold value to determine the degree of association required to consider a data point as part of a cluster.

        Returns:
        - coverage: Boolean matrix indicating whether each data point is covered by each cluster.
        """
        coverage = np.zeros((X.shape[0], k), dtype=bool)
        for instance in range(fuzzy_matrix.shape[0]):
            for cluster_num in range(fuzzy_matrix.shape[1]):
                if fuzzy_matrix[instance][cluster_num] > threshold:
                    coverage[instance][cluster_num] = True
        return coverage

    def Fuzzy_K_Means(self, X, reuters_tags, k=10, initializer=None, threshold=0.10):
        """
        Perform Fuzzy K-Means clustering on input data and calculate the F1-measure using Reuters tags.

        Args:
        - X: Input data to cluster.
        - reuters_tags: Reuters tags for the input data.
        - k: Number of clusters to form.
        - initializer: Method for initialization.
        - threshold: Threshold value to determine the degree of association required to consider a data point as part of a cluster.

        Returns:
        - precision: Precision of the clustering.
        - recall: Recall of the clustering.
        - F1_Measure: F1-measure of the clustering.
        """
        fuzzy_matrix = self.Fuzzy_K_Means_(X, k, initializer)
        coverage = self.get_Fuzzy_K_Means_coverage(fuzzy_matrix, X, k, threshold)
        precision, recall, F1_Measure = self.f1_measure(coverage, reuters_tags)
        return precision, recall, F1_Measure

    #################################################################################
    #################################################################################
    #################################################################################
    #                                       OKM                                     #
    #################################################################################
    #################################################################################
    #################################################################################

    #################################################################################
    #                               AUXILIAR METHODS                                #
    #################################################################################

    def check_prototype(self, prototype, avoid):
        '''
        Checks if a prototype is in the avoid list or not.

        Args:
        prototype (array): The prototype to be checked.
        avoid (list): The list of prototypes to avoid.

        Returns:
        bool: True if prototype is in avoid list, False otherwise.
        '''
        for elem in avoid:
            if np.array_equal(prototype, elem):
                return True
        return False

    def phi(self, mc):
        '''
        Computes the mean of a collection of vectors.

        Args:
        mc (array): The collection of vectors.

        Returns:
        array: The mean vector.
        '''
        return np.sum(mc, axis=0) / len(mc)

    def nearest_prototype(self, instance, prototypes, avoid):
        '''
        Finds the nearest prototype for a given instance among a set of prototypes.

        Args:
        instance (array): The instance for which the nearest prototype is to be found.
        prototypes (list): The set of prototypes to choose from.
        avoid (list): The prototypes to avoid.

        Returns:
        tuple: A tuple containing the closest prototype and its index in the prototypes list.
        '''
        min_dist = float('inf')
        closest_prototype = None
        min_id = 0
        id = 0
        for prototype in prototypes:
            if self.check_prototype(prototype, avoid):
                continue
            instance_arr = np.array(instance)
            prototype_arr = np.array(prototype)
            dist = math.sqrt(np.sum((instance_arr - prototype_arr) ** 2))
            if dist < min_dist:
                min_dist = dist
                closest_prototype = prototype
                min_id = id
            id = id + 1
        return closest_prototype, min_id

    def PROTOTYPE(self, X, k, prototypes, point_prototype_pair):
        '''
        Calculates the prototype vector for a given set of instances.

        Args:
        X (array): The set of instances.
        k (int): The number of prototypes.
        prototypes (list): The list of prototypes.
        point_prototype_pair (list): The mapping between instances and their prototypes.

        Returns:
        array: The prototype vector.
        '''
        numerator_sum = [0] * k
        denominator_sum = [0] * k
        for id in range(X.shape[0]):
            alpha = 1 / (len(point_prototype_pair[id]) ** 2)
            m_h = X[id][:] * len(point_prototype_pair[id]) - np.sum(point_prototype_pair[id], axis=0)
            for i in range(len(prototypes)):
                if self.check_prototype(prototypes[i], point_prototype_pair[id]):
                    numerator_sum[i] = numerator_sum[i] + (alpha * m_h)
                    denominator_sum[i] = denominator_sum[i] + alpha
        return [num / den for num, den in zip(numerator_sum, denominator_sum)]

    def objective_function(self, X, point_prototype_pair):
        '''
        Calculates the value of the objective function for a given set of instances.

        Args:
        X (array): The set of instances.
        point_prototype_pair (list): The mapping between instances and their prototypes.

        Returns:
        float: The value of the objective function.
        '''
        return sum([math.sqrt(np.sum((X[id] - self.phi(point_prototype_pair[id])) ** 2)) for id in range(len(X))])

    #################################################################################
    #                               MAIN METHODS                                    #
    #################################################################################

    def ASSIGN(self, instance, prototypes, A_i_old=None, A_i_indices_old=None):
        """
        Assigns an instance to a prototype and updates phi and id.
        Args:
        - instance: an array representing an instance
        - prototypes: an array containing the current prototype
        - A_i_old: (optional) an array representing the previous prototype of an instance
        - A_i_indices_old: (optional) an array representing the indices of the previous prototype of an instance

        Returns:
        - instance_prototypes: an array representing the prototype to which each instance is assigned.
        - instance_prototypes_indices: an array representing the indices of the prototype to which each instance is assigned.
        """
        instance_prototypes = []
        instance_prototypes_indices = []
        phi_values = []
        id = -1

        while len(instance_prototypes) <= len(prototypes):

            # Step 1 & 2:
            closest_prototype, min_id = self.nearest_prototype(instance, prototypes, instance_prototypes)
            if closest_prototype is not None:
                instance_prototypes.append(closest_prototype)
                instance_prototypes_indices.append(min_id)
                phi_values.append(self.phi(instance_prototypes))
                id = id + 1
            if id == 0: continue

            # Step 3:
            prev_distance = math.sqrt(np.sum((instance - phi_values[id - 1]) ** 2))
            current_distance = math.sqrt(np.sum((instance - phi_values[id]) ** 2))
            if (current_distance < prev_distance): continue
            else:
                if A_i_old == None and A_i_indices_old == None: return instance_prototypes[:id], instance_prototypes_indices[:id]
                old_distance = math.sqrt(np.sum((instance - self.phi(A_i_old)) ** 2))
                if prev_distance <= old_distance: return instance_prototypes[:id], instance_prototypes_indices[:id]
                else: return A_i_old, A_i_indices_old

    def okm_(self, X, k, t_max, epsilon, initializer):
        """
        Performs OKM clustering on the input data.
        Args:
        - X: an array of input data points
        - k: an integer representing the number of clusters
        - t_max: an integer representing the maximum number of iterations
        - epsilon: a small float value representing the tolerance for convergence
        - initializer: an array representing the initial cluster prototypes

        Returns:
        - labels: an array of indices indicating the cluster to which each input point belongs
        """
        all_prototypes = []

        # Step 1: Initialize k cluster prototypes randomly
        prototypes = initializer
        all_prototypes.append(prototypes)

        # Step 2: Compute assign for each point to get the coverage
        point_prototype_pair = []
        labels = []
        for instance in X:
            pair, indice = self.ASSIGN(instance, prototypes, None, None)
            point_prototype_pair.append(pair)
            labels.append(indice)
        J_old = self.objective_function(X, point_prototype_pair)

        # Step 3: Set t=0
        t = 0

        while t < t_max:

            # Step 4: Compute the new prototype for each cluster
            final_m_h = self.PROTOTYPE(X, k, prototypes, point_prototype_pair)
            all_prototypes.append(final_m_h)

            # Step 5: Compute new assign for each point to get the coverage
            old_point_prototype_pair = point_prototype_pair
            old_point_prototype_indices_pair = labels
            point_prototype_pair = []
            labels = []
            id = 0
            for instance in X:
                pair, indice = self.ASSIGN(instance, final_m_h, old_point_prototype_pair[id], old_point_prototype_indices_pair[id])
                point_prototype_pair.append(pair)
                labels.append(indice)
                id = id + 1

            # Step 6: Check convergence
            J = self.objective_function(X, point_prototype_pair)
            if J_old - J > epsilon:
                J_old = J
                t = t + 1
            else: break
        return labels

    def get_OKM_coverage(self, labels, X, k):
        """
        Calculates the coverage matrix of instances and prototypes

        Args:
            labels (list): List of tuples containing the prototypes assigned to each instance
            X (numpy array): Input data
            k (int): Number of prototypes

        Returns:
            numpy array: Coverage matrix of instances and prototypes
        """
        coverage = np.zeros((X.shape[0], k), dtype=bool)
        for index in range(len(labels)):
            for p in labels[index]:
                coverage[index][p] = True
        return coverage

    def okm(self, X, reuters_tags, k=10, t_max=100, epsilon=1e-6, initializer=None):
        """
        Runs Optimum-Path Forest Algorithm on input data.

        Args:
            X (numpy array): Input data
            reuters_tags (numpy array): Array containing the tags of each instance
            k (int): Number of prototypes (default 10)
            t_max (int): Maximum number of iterations (default 100)
            epsilon (float): Tolerance for convergence (default 1e-6)
            initializer (numpy array): Array containing the initial prototypes (default None)

        Returns:
            tuple: Tuple containing the precision, recall, and F1-measure scores
        """
        labels = self.okm_(X, k, t_max, epsilon, initializer)
        coverage = self.get_OKM_coverage(labels, X, k)
        precision, recall, F1_Measure = self.f1_measure(coverage, reuters_tags)
        return precision, recall, F1_Measure