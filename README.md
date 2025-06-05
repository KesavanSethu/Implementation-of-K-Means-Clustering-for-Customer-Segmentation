# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries and load the dataset (Mall_Customers.csv) using pandas.
2. Display dataset structure, first few rows, and check for missing values.
3. Use the Elbow Method by fitting KMeans with 1 to 10 clusters and plotting WCSS to find the optimal number.
4. Apply KMeans clustering with the chosen number of clusters (e.g., 5) to the relevant features.
5. Predict cluster labels for each customer and add them to the dataset as a new column.
6. Filter the dataset into separate DataFrames for each cluster.
7. Plot each cluster using a scatter plot to visualize customer segments based on income and spending.
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Kesavan S 
RegisterNumber:  212224230121
*/
```
```
# importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
```
```
# load the dataset
data = pd.read_csv('Mall_Customers.csv')
```
```
# display first five records
data.head()
```
```
# display dataframe information
data.info()
```
```
# display the count of null values in each column
data.isnull().sum()
```
```
# using Elbow Method to determine optimal number of clusters
from sklearn.cluster import KMeans
# within-cluster sum of squares
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++")
    # using 'Annual Income' and 'Spending Score'
    kmeans.fit(data.iloc[:, 3:])
    wcss.append(kmeans.inertia_)

# plot the Elbow graph
plt.plot(range(1, 11), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()
```

```
# fitting KMeans to the data with 5 clusters
km = KMeans(n_clusters=5)
km.fit(data.iloc[:, 3:])
```
```
# predicting cluster labels for each record
y_pred = km.predict(data.iloc[:, 3:])
y_pred
```
```
# adding predicted cluster labels to the dataframe
data["cluster"] = y_pred
```
```
# splitting data into clusters for visualization
df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]
```
```
# visualizing customer segments using scatter plot
plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c="red", label="cluster0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c="black", label="cluster1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c="blue", label="cluster2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c="green", label="cluster3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c="magenta", label="cluster4")
plt.legend()
plt.title("Customer Segments")
plt.show()
```

## Output:

**Head Values**

![Screenshot 2025-05-12 183016](https://github.com/user-attachments/assets/d86d64d3-acbf-4431-9d50-424f5b5bf133)


**Dataframe Info**

![Screenshot 2025-05-12 183021](https://github.com/user-attachments/assets/ec1f15cd-0910-4b27-a220-47deb0354f9a)

**Sum - Null Values**

![Screenshot 2025-05-12 183026](https://github.com/user-attachments/assets/2b09fd82-303d-4086-9667-fc55df71f11c)

**Elbow Graph**

![Screenshot 2025-05-12 183039](https://github.com/user-attachments/assets/f0f34382-0f93-4c75-9e32-e8993ca38f04)

**Training the model**

![Screenshot 2025-05-12 183043](https://github.com/user-attachments/assets/39821292-7dba-4017-bb7f-e7cd202572e0)

**Predicting cluster Labels**

![Screenshot 2025-05-12 183048](https://github.com/user-attachments/assets/ffea1c53-4665-4d5f-963c-5867f82f3bb1)

**Visualizing Customer Labels**

![Screenshot 2025-05-12 183057](https://github.com/user-attachments/assets/8d9a8a4d-323e-4dfd-9957-bfef9e906682)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
