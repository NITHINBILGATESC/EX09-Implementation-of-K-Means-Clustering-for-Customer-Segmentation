# EX 9 Implementation of K Means Clustering for Customer Segmentation
## DATE:
## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. This iterative process helps K-Means identify distinct clusters of data points. 
2. Before you apply the algorithm, you need a dataset with relevant features. 
3. Once you determine the optimal number of clusters 𝐾, you can apply the algorithm to segment the customers. 
4. If you're using 2D data or select two features for simplicity 
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: NITHIN BILGATES C
RegisterNumber:  2305001022
*/
```import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

data=pd.read_csv("/content/Mall_Customers_EX8 (1).csv")
data

data = data[['Annual Income (k$)', 'Spending Score (1-100)']]

plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show

k=3

Kmeans = KMeans(n_clusters=k)
Kmeans.fit(data)

centroids = Kmeans.cluster_centers_

labels = Kmeans.labels_
print("centroids:")
print(centroids)
print("labels:")
print(labels)

colors = ['r', 'g', 'b']
for i in range(k):
    cluster_data = data[labels == i]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], c=colors[i])
    distance = euclidean_distances(cluster_data, [centroids[i]])
    radius = np.max(distance)
    circle = plt.Circle(centroids[i], radius, color=colors[i], fill=False)
    plt.gca().add_patch(circle)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='k', label='Centroids')
plt.title('k-means clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

```
## Output:
![image](https://github.com/user-attachments/assets/8b8df5cb-66bd-4570-b7eb-50a70c7bd999)
![image](https://github.com/user-attachments/assets/ad90562b-9a9f-4af4-a1aa-294416c94527)
![image](https://github.com/user-attachments/assets/b2578f88-a634-4447-8d1a-53afdb61d5ae)
![image](https://github.com/user-attachments/assets/cbd20632-1d15-446c-bae6-b1ed7ed0c429)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
