# NICASN
Discovering clusters in social networks is of fundamental and practical interest. This paper presents a novel clustering strategy for large-scale highly-connected social net- works. We propose a new hybrid clustering technique based on non-negative matrix fac- torization and independent component analysis for finding complex relationships among users of a large-scale network. We extract the important features of the network and then perform clustering on independent and important components of the network. Above this, we introduce a new k-means centroid initialization method by which we achieve higher performance. We apply our approach on four well-known social networks: Face- book, Twitter, Academia and Youtube. The experimental results show that our approach generally achieves better results in terms of the Silhouette coefficient compared to well- known clustering methods such as Hierarchical Louvain, Multiple Local Community detection, and k-means++. In general, our approach outperforms the state-of-the-art techniques when dealing with complex and highly-connected networks

To install NICASN try these steps:
1- git clone https://github.com/humanworth/NICASN
2- cd to clone directory
3- run pip install -r /path/to/requirements.txt
4- For each dataset run python NICASN-@dataset > @dataset.log (where @dataset is the corresponding dataset)

Comment if you have further questions
