# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 21:31:59 2021

@author: Ali
"""


from NetworkClustering import KMeansClustering
from sklearn.model_selection import GridSearchCV
from sknetwork.data import load_edge_list
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import numpy as np

def Print_Silhouete_score(n_clusters=2,X=[[2,3],[4,5]],score_value=1,sample_silhouette_values=[2,3,4],cluster_labels=[1,1,1,0,0,0],centers=None):

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=score_value, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    #centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

   
    
    plt.savefig(fname= n_clusters.__str__())
    #plt.show()
    return plt




graph = load_edge_list('dataset/socfb-Stanford3.mtx')
adjacency = graph.adjacency
names = graph.names
#k = 1000  #Number of Users of the dataset
cv = [(slice(None), slice(None))] #to remove cross validation
#adjacency = adjacency[:k][:,:k]
est=KMeansClustering(n_clusters=2, n_neighbors=10, n_components=50, adjacency=adjacency)
parameters = {'n_clusters':[14],'n_neighbors':[5],'n_components':[14,50,60,70,80,90,100],'adjacency':[adjacency]}
clf = GridSearchCV(est, parameters,error_score='raise',cv=cv)
adjacency._shape[0]
clf.fit(adjacency)
#sorted(clf.cv_results_.keys())
result=clf.cv_results_

if result is not None:
    for i in range(0,len(result["params"])):
        print("--------------For No. of Clusters = %s" %result['param_n_clusters'][i],
              " and No. of Components = %s" %result['param_n_components'][i],
              "----> Ranking = %s" %result["rank_test_score"][i])
        
        print("Mean Test Score = %s" %result['mean_test_score'][i], "--- Mean Fit Time = %s" %result['mean_fit_time'][i], "--- Mean Score Time = %s" %result["mean_score_time"][i])
        print("STD Fit Time = %s" %result['std_fit_time'][i], "--- STD Score Time = %s" %result['std_score_time'][i], "--- STD Test Score = %s" %result["std_test_score"][i])
        print("\n")

best_score=clf.best_score_
best_params=clf.best_params_
best_index=clf.best_index_

print("Best Parameters -> Clusters= %s" %best_params["n_clusters"], " Components = %s" %best_params["n_components"], 
      "Best Index = %s" %best_index, "Best Score= %s" %best_score )

labels=est.return_label(adjacency,n_clusters=best_params["n_clusters"],n_components=best_params["n_components"])
transform_result=est.transform_result
est.bestscore




import matplotlib.pyplot as plt
clusters=best_params["n_clusters"]
range_n_clusters=[clusters]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    cluster_labels = labels
    trans=transform_result
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(trans) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
   # clusterer = KMeans(n_clusters=n_clusters)

    centers=est.centers
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = est.bestscore #silhouette_score(adjacency, cluster_labels)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(trans,cluster_labels)
    plot=Print_Silhouete_score(n_clusters,trans,silhouette_avg,sample_silhouette_values,cluster_labels=cluster_labels,centers=centers)
    #plot.savefig(fname="avgsilhoutte", format='pdf')


















def show2D(data,label):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = plt.axes()
          # Data for a three-dimensional line
    xline = data[:,0]
    yline = data[:,1]
       # ax.plot(xline, yline 'gray')
    
        # Data for three-dimensional scattered points
    ax.scatter(xline, yline,c=label, s=2);
    return fig
























