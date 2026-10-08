# Customer Segmentation (K-means + Hierarchical Clustering)

I grouped 200 mall customers into segments using only their age, income and spending score. There are no labels in this data, so the task is unsupervised: I never told the model what a "good customer" looks like, it had to find the groups on its own.

I picked K-means with 6 clusters and then checked the result against hierarchical clustering. The two methods put 191 of the 200 customers in matching groups.

## Data

[Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) from Kaggle. 200 customers, with `CustomerID`, `Gender`, `Age`, `Annual Income (k$)` and `Spending Score (1-100)`.

There were no missing values and no duplicate rows. I dropped `CustomerID` (it's just a counter) and encoded `Gender` as 0/1. I left Gender out of the clustering itself and used Age, Annual Income and Spending Score. I didn't test whether adding it changes the groups.

## Approach

1. **Scale the features** with `StandardScaler`. K-means works on distances, so without scaling the feature with the biggest numbers would dominate.
2. **Check PCA.** The cumulative explained variance was 44%, 78% and 100% for PC1, PC2 and PC3. The variance is spread fairly evenly, so PCA couldn't shrink 3 features without losing real information (dropping PC3 would lose about 22%). I clustered on the scaled features directly. PC1 is mostly an age-versus-spending axis, and PC2 is almost entirely income.
3. **Choose K** with the elbow method and the silhouette score.
4. **Run K-means** and profile each cluster using the average age, income and spending score.
5. **Cross-check with hierarchical clustering** (Ward linkage) and compare the two sets of groups.

## Why 6 clusters

I started with K = 4. The inertia curve bends around there: going from K = 3 to 4 dropped inertia by about 90, while 4 to 5 dropped it by only about 37. But the bend is gentle, so the elbow alone couldn't settle it.

The silhouette score was highest at K = 6 (0.428), with K = 4 at 0.404. That is a small gap, and every K from 4 to 8 scored within about 0.02 of each other, so the score alone doesn't prove much. K = 2 and K = 3 were clearly worse (0.335 and 0.358).

The dendrogram gave a third opinion. Cutting the tree at a merge distance of 8 gives 6 groups.

What decided it for me was comparing the actual groups. At K = 4, one cluster averaged 25 years old and 40k income, but it was mixing two kinds of customers. K = 6 splits that cluster into:

- young customers with low income (about 25k) who spend a lot
- slightly older customers with middle income (about 57k) who spend an average amount

Those are different customers, and K = 4 hid the difference. K = 6 also pulled out a low-income, low-spending group that K = 4 had folded into an older cluster.

## The 6 segments

Averages from the original (unscaled) data:

| Cluster | Customers | Avg age | Avg income (k$) | Avg spending score | Description |
|---|---|---|---|---|---|
| 0 | 45 | 56 | 54 | 49 | Older, middle income, middle spenders |
| 1 | 39 | 27 | 57 | 48 | Younger, middle income, middle spenders |
| 2 | 33 | 42 | 89 | 17 | High income, low spenders |
| 3 | 39 | 33 | 87 | 82 | High income, high spenders |
| 4 | 23 | 25 | 25 | 78 | Young, low income, high spenders |
| 5 | 21 | 46 | 26 | 19 | Low income, low spenders |

Clusters 2 and 3 earn about the same and behave in opposite ways. A mall would probably want to treat them differently, for example by targeting cluster 2 with promotions to get them spending.

## Cross-check with hierarchical clustering

I cut the Ward dendrogram into 6 groups and compared them to the K-means clusters with a crosstab. The group sizes were close (K-means: 45, 39, 33, 39, 23, 21; dendrogram: 45, 45, 39, 28, 22, 21), and 191 of 200 customers landed in matching groups. Three clusters matched perfectly. The 9 customers that differ sit on the borders between neighbouring groups, mostly around cluster 2.

Clusters 0 and 1 have almost the same income and spending. Only age separates them (56 vs 27). Both methods still split them cleanly, which suggests the age difference is real and not just K-means cutting one blob in half.

## Limitations

- The dataset is small (200 customers) and has only three usable features, so the segments are simple.
- The choice of K is a judgment call. The silhouette scores for K = 4 to 8 are close, and I chose 6 partly because the groups made the most sense when I described them.
- No single 2D plot shows all six clusters cleanly. On the income vs spending plot, clusters 0 and 1 blend together because age, the feature that separates them, isn't on the axes.
- I didn't test whether including Gender changes the clusters.

## Plots

![Elbow plot](images/elbow.png)
![Dendrogram](images/dendrogram.png)
![Clusters: income vs spending](images/clusters_income_spending.png)

## Tools

Python, pandas, NumPy, scikit-learn, SciPy, Matplotlib.