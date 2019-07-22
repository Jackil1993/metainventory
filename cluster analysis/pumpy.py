''' This script is written to conduct a case-study reported in the paper
"Unsupervised learning-based SKU segmentation.
The script utilizes a free software machine learning library “scikit-learn” as a core
complementing it with several algorithms.
The script uses the concept of data-pipeline to consequentially perform the following procedures:
to impute the missing data with nearest-neighbour-imputation
to standardize the data
to identify and trim outliers and small 'blobs' with LOF and mean-shift
to cluster the data with k-mean and DBSCAN
to improve the eventual clustering result via PCA
Since the ground truth is not provided, the clustering is validated only by internal evaluation, namely
by silhouette index, Calinski-Harabazs index and Dunn-index '''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from fancyimpute import KNN
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift, KMeans, DBSCAN, estimate_bandwidth
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import FactorAnalysis
from scipy import ndimage



class Pipeline:

    def __init__(self, methods):
        self.methods = methods

    def pump(self):
        for method in self.methods:
            method
        w = Writer(pd.DataFrame(p.data))
        w.write_to_excel()


class Processing:

    def __init__(self, data, k=10, n_neighbors=20):
        self.data = data
        self.k = k
        self.n_neighbors = n_neighbors

    def knn_imputation(self):
        self.data = pd.DataFrame(KNN(self.k).fit_transform(self.data))

    def standardization(self):
        self.data = preprocessing.scale(self.data)

    def local_outlier_factor(self, drop_anomalies=True):
        lof = LocalOutlierFactor(self.n_neighbors)
        predicted = lof.fit_predict(self.data)

        data_with_outliers = pd.DataFrame(self.data)
        data_with_outliers['outliers'] = pd.Series(predicted, index=data_with_outliers.index)

        if drop_anomalies is True:
            def drop_outliers(data):
                data = data_with_outliers
                data = data.sort_values(by=['outliers'])
                outliers_number = -data[data.outliers == -1].sum().loc['outliers'].astype(int)
                print(outliers_number, " outliers are found")
                return data, data.iloc[outliers_number:], outliers_number

            data_with_outliers, data_without_outliers, outliers_number = drop_outliers(data_with_outliers)

        w = Writer(data_without_outliers, sheet='Sheet2', file='outliers.xlsx')
        w.write_to_excel()

    def get_data(self):
        return self.data


class Reduction:

    def __init__(self, n_components=2):
        self.n_components = n_components

    def pca(self, data):
        compressor = PCA(self.n_components)
        compressor.fit(data)
        return compressor.transform(data), compressor.explained_variance_ratio_.sum()

    def factor_analysis(self, data):
        def ortho_rotation(lam, method='varimax', gamma=None, eps=1e-6, itermax=100):

            if gamma == None:
                if (method == 'varimax'):
                    gamma = 1.0

                nrow, ncol = lam.shape
                R = np.eye(ncol)
                var = 0

                for i in range(itermax):
                    lam_rot = np.dot(lam, R)
                    tmp = np.diag(np.sum(lam_rot ** 2, axis=0)) / nrow * gamma
                    u, s, v = np.linalg.svd(np.dot(lam.T, lam_rot ** 3 - np.dot(lam_rot, tmp)))
                    R = np.dot(u, v)
                    var_new = np.sum(s)
                    if var_new < var * (1 + eps):
                        break
                    var = var_new
            print(var)
            print(R)
            return R


        transformer = FactorAnalysis(n_components=self.n_components, random_state=0)
        transformed_data = transformer.fit_transform(data)
        r = ortho_rotation(transformed_data)
        #r = np.array([[0.96,0.279],[-0.279,0.96]])
        #r = np.array([[0.408, -0.91], [0.91, 0.408]])
        transformed_data = np.matmul(r, np.transpose(transformed_data))
        #corr = pd.DataFrame(np.concatenate((data.values, transformed_data), axis=1)).corr()

        #w = Writer(corr, sheet='Sheet2', file='corr.xlsx')
        #w.write_to_excel()
        return transformed_data


class Clustering:

    def __init__(self, data):
        self.data = data

    def mean_shift_clustering(self, plot=False, drop_small_clusters=True, threshold=4):

        def shift(data):
            bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=500)
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(data)
            return ms.labels_, ms.cluster_centers_, pd.DataFrame(data)

        labels, cluster_centers, labeled_data = shift(self.data)

        old_id = labeled_data.index
        iteration = 1
        while drop_small_clusters is True and iteration<4:
            dropped = 0
            labeled_data['mean-shift'] = pd.Series(labels)
            labeled_data['clusters_sorted'] = pd.Series(labels).value_counts()

            to_drop = []
            labeled_data.reset_index(drop=True)

            for cluster in labeled_data.index:

                if labeled_data.loc[cluster, 'clusters_sorted']< threshold:

                    for row in range(0, len(labeled_data['mean-shift'])):
                        if labeled_data.loc[row, 'mean-shift'] == cluster:
                            to_drop.append(row)

            for i in to_drop:
                labeled_data = labeled_data.drop(i)
                dropped += 1

            labeled_data = labeled_data.drop(['clusters_sorted', 'mean-shift'], axis=1)
            labels, cluster_centers, labeled_data = shift(labeled_data)
            iteration += 1
            print("iteration: ", iteration, "clusters: ", max(labels)+1, "dropped: ", dropped)


        labeled_data['mean-shift'] = pd.Series(labels)
        labeled_data['clusters_sorted'] = pd.Series(labels).value_counts()
        w = Writer(labeled_data, sheet='Sheet2', file='labeled.xlsx')
        w.write_to_excel()

        labeled_data = labeled_data.drop('mean-shift', axis=1)
        labeled_data = labeled_data.drop('clusters_sorted', axis=1)
        self.data = labeled_data
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        print("mean-shift ", '\n', "number of estimated clusters: {} ".format(n_clusters_))
        e = Evaluation(self.data, labels)
        e.evaluate()

        if plot is True:
            red = Reduction(n_components=2)
            to_plot, variance = red.pca(labeled_data)
            plotter = Plotter()
            plotter.plot_clustering(to_plot, n_clusters_, labels, cluster_centers, variance, 'mean-shift')

    def k_means_clustering(self, k, plot_best=True, compress=0, method='pca'):
        if compress != 0:
            if method=='pca':
                r = Reduction(compress)
                self.data, variance = r.pca(self.data)
                print(variance)
            else:
                r = Reduction(compress)
                self.data = r.factor_analysis(self.data)

        km = KMeans(n_clusters=k, random_state=0).fit(self.data)
        labels = km.labels_
        cluster_centers = km.cluster_centers_

        if plot_best is True:
            red = Reduction(n_components=2)
            to_plot, variance = red.pca(self.data)
            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            plotter = Plotter()
            plotter.plot_clustering(to_plot, n_clusters_, labels, cluster_centers, variance, 'K-means')

        e = Evaluation(self.data, labels)
        print(k, '-means')
        e.evaluate()

    def dbscan(self, eps, plot_best=False):
        db = DBSCAN(eps=eps, min_samples=10).fit(self.data)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        n_clusters_ = len(set(labels))

        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        e = Evaluation(self.data, labels)
        print('DBSCAN ', 'eps=', eps, '\n', n_clusters_, ' clusters found')
        print(list(labels).count(-1), " observations considered as noise")
        e.evaluate()

        if plot_best is True:
            red = Reduction(n_components=2)
            to_plot, variance = red.pca(self.data)
            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            plotter = Plotter()
            plotter.plot_clustering(to_plot, n_clusters_, labels, np.nan, variance, 'DBSCAN')


class Evaluation:

    def __init__(self, data, labels, metric='euclidean'):
        self.data = data
        self.labels = labels
        self.metric = metric

    def silhouette(self):
        return metrics.silhouette_score(self.data, self.labels, metric=self.metric)

    def calinski_harabaz(self):
        return metrics.calinski_harabaz_score(self.data, self.labels)

    def dunn_index(self):
        def normalize_to_smallest_integers(labels):
            max_v = len(set(labels))

            sorted_labels = np.sort(np.unique(labels))
            unique_labels = range(max_v)
            new_c = np.zeros(len(labels), dtype=np.int32)
            for i, clust in enumerate(sorted_labels):
                new_c[labels == clust] = unique_labels[i]
            return new_c

        def dunn(labels, distances):
            labels = normalize_to_smallest_integers(labels)

            unique_cluster_distances = np.unique(min_cluster_distances(labels, distances))
            max_diameter = max(diameter(labels, distances))

            if np.size(unique_cluster_distances) > 1:
                return unique_cluster_distances[1] / max_diameter
            else:
                return unique_cluster_distances[0] / max_diameter

        def min_cluster_distances(labels, distances):
            labels = normalize_to_smallest_integers(labels)
            n_unique_labels = len(np.unique(labels))

            min_distances = np.zeros((n_unique_labels, n_unique_labels))
            for i in np.arange(0, len(labels) - 1):
                for ii in np.arange(i + 1, len(labels)):
                    if labels[i] != labels[ii] and distances[i, ii] > min_distances[labels[i], labels[ii]]:
                        min_distances[labels[i], labels[ii]] = min_distances[labels[ii], labels[i]] = distances[i, ii]
            return min_distances

        def diameter(labels, distances):
            labels = normalize_to_smallest_integers(labels)
            n_clusters = len(np.unique(labels))
            diameters = np.zeros(n_clusters)

            for i in np.arange(0, len(labels) - 1):
                for ii in np.arange(i + 1, len(labels)):
                    if labels[i] == labels[ii] and distances[i, ii] > diameters[labels[i]]:
                        diameters[labels[i]] = distances[i, ii]
            return diameters

        return dunn(self.labels, euclidean_distances(self.data))

    def evaluate(self):
        coeff = ['Silhouette: ', self.silhouette(), 'Calinski-Harabaz: ',
                 self.calinski_harabaz(), 'Dunn: ', self.dunn_index()]
        print(coeff)


class Plotter:

    def plot_clustering(self, data, n_clusters_, labels, cluster_centers, variance, name):
        plt.figure()
        plt.rc('font', size=14)
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        if -1 in labels:
            for k, col in zip(range(-1, n_clusters_-1), colors):
                my_members = labels == k
                if k==-1:
                    plt.plot(data[my_members, 0], data[my_members, 1], col + '+')
                else:
                    plt.plot(data[my_members, 0], data[my_members, 1], col + '.')
        else:
            for k, col in zip(range(n_clusters_), colors):
                my_members = labels == k
                plt.plot(data[my_members, 0], data[my_members, 1], col + '.')
        plt.title('Algoritm: {} Number of clusters: {}. \n'
                  '{}% of variance is preserved after PCA'.format(name, n_clusters_, round(variance*100, 2)))
        plt.show()


class Writer:

    def __init__(self, data, sheet='Sheet1', file='new_data.xlsx'):
        self.data = data
        self.sheet = sheet
        self.file = file
        self.writer = pd.ExcelWriter(self.file, engine='xlsxwriter')

    def write_to_excel(self):
        self.data.to_excel(self.writer, sheet_name=self.sheet)
        self.writer.save()


file = pd.ExcelFile("initial_data.xlsx")
data = file.parse("Sheet1")
data[['Unitprice', 'Expire date', 'Pal grossweight', 'Pal height',
      'Units per pal']] = data[['Unitprice', 'Expire date', 'Pal grossweight',
                                'Pal height', 'Units per pal']].replace(0.0, np.nan)
data = data.drop(["ID",	"Tradability",	"Init status"], axis=1)


p = Processing(data, 10)
preprocessing_methods = [p.knn_imputation(), p.standardization(), p.local_outlier_factor(drop_anomalies=True)]
pipe1 = Pipeline(preprocessing_methods)
pipe1.pump()
c = Clustering(p.get_data())
pipe2 = Pipeline([c.mean_shift_clustering(), c.k_means_clustering(10, plot_best=True, compress=2, method='factor')])
pipe2.pump()
