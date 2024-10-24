# EX 9 Implementation of K Means Clustering for Customer Segmentation
## DATE:
## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.This iterative process helps K-Means identify distinct clusters of data points. 
2.Before you apply the algorithm, you need a dataset with relevant features. 
3.Once you determine the optimal number of clusters 𝐾, you can apply the algorithm to segment the customers. 
4.If you're using 2D data or select two features for simplicity 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: NITHIN BILGATES C
RegisterNumber: 2305001022 
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

k = 3

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
![image](https://github.com/user-attachments/assets/ce08247d-c260-4100-b0e6-68a0a5f0993c)
![image](https://github.com/user-attachments/assets/00878588-43cd-4c71-a667-e0d3f4b26b40)
![image](https://github.com/user-attachments/assets/6e1f5dbe-1401-4b5a-8a1f-46d582a7fdf2)
![image](https://github.com/user-attachments/assets/c1f81d10-91bb-4fec-b921-fc6322929a42)
![image](https://github.com/user-attachments/assets/a6e64fe0-2ddb-412b-a507-46f19ce5367e)




## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
