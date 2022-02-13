# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 10:00:26 2021

@author: Ali
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
############# Extra classes for estimator
from sklearn import manifold#, datasets
from sklearn.decomposition import  PCA,FastICA,NMF
import matplotlib.pyplot as plt
from sknetwork.embedding import cosine_modularity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.preprocessing import Normalizer
from sknetwork.clustering import modularity
np.random.RandomState(seed=1000)
class KMeansClustering(BaseEstimator):
    def __init__(self, n_clusters=10,n_neighbors=10,n_components=50,adjacency=None):
        self.n_clusters = n_clusters
        self.n_neighbors=n_neighbors
        self.n_components=n_components
        self.transform_result=None
        self.adjacency=adjacency
        self.label=None
        self.bestscore=None
        self.centers=None
        self.H=None
        
    def show2D(data,label):
        fig = plt.figure()
        ax = plt.axes()
        # Data for a three-dimensional line
        xline = data[:,0]
        yline = data[:,1]
        #ax.plot(xline, yline 'gray')
    
        # Data for three-dimensional scattered points
        ax.scatter(xline, yline,c=label, s=2);
        return fig

    
    def _transform(self, X):
       if self.n_clusters>self.n_components:
           print("the number of clusters must be lower than the number of components")
           
       assert self.n_clusters<=self.n_components
       nmf=NMF(n_components=self.n_components,init='nndsvd')
       trans= nmf.fit_transform(X)
       H = nmf.components_
       ica = FastICA(n_components=None)
       Y_ICA = ica.fit(trans).transform(trans)  # Estimate the 
       modularity=cosine_modularity(self.adjacency, Y_ICA, weights='degree')
       print("For components =%s" % self.n_components, ", Cosine Modularity ICA = %s" % modularity)
       modularity=cosine_modularity(X, trans, weights='degree')
       print("For components =%s" % self.n_components, ", Cosine Modularity NMF = %s" % modularity)
       self.H=H[:self.n_clusters,:self.n_components] #transh[:self.n_clusters,:self.n_components]
       self.transform_result=Y_ICA
       return Y_ICA
    
    def transform(self, adjacency):
        return self._transform(adjacency)
    
    def predict(self, adjacency):
        transit=self.transform(adjacency=adjacency)
        clusterer = KMeans(n_clusters=self.n_clusters,init=self.H,random_state=np.random.RandomState(seed=1000))
        labels=clusterer.fit_predict(transit)
        return labels
       
    def fit(self, adjacency):
        # Return the classifier
        print("----------------------- PRINT N_CLUSTERS --> %s" %self.n_clusters)
        print("----------------------- PRINT N_components  --> %s" %self.n_components)

        if self.adjacency is None:
            self.adjacency=adjacency
        self.label=self.predict(self.adjacency)
        mod=modularity(adjacency, self.label) 
        print("----------------------- PRINT Modularity  --> %s" %mod)
        return self
    
    def fit_transform(self, adjacency):
       return self.fit(adjacency).transform(adjacency)
    
    def return_label(self, adjacency,n_clusters=2,n_neighbors=10,n_components=50):
        self.n_neighbors=n_neighbors
        self.n_components=n_components
        self.n_clusters=n_clusters
        self.transit=self.transform(adjacency=adjacency)
        clusterer = KMeans(n_clusters=n_clusters)
        labels=clusterer.fit_predict(self.transit) 
        self.bestscore=round(silhouette_score(self.adjacency,labels),10)
        self.centers=clusterer.cluster_centers_
        return labels
    
    def fit_predict(self, adjacency):
    
        return self.fit(adjacency).predict(adjacency)
    
    def score(self, adjacency=None):
        print("------------- Scoring  ---------------------------------")
        return  round(silhouette_score(self.adjacency,self.label),10)
    





 