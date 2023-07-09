#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[2]:


import pandas as pd


# In[8]:


data = pd.read_csv(r"C:\Users\sonas\Downloads\mcdonalds.csv")


# In[9]:


data.head()


# In[10]:


data.info()


# In[7]:


data.head()


# In[11]:


data.describe()


# In[12]:


import numpy as np


# In[16]:


data.shape


# In[18]:


column_names = data.columns
print(column_names)


# In[19]:


data.head(3)


# In[20]:


MD_x = data.iloc[:, 0:11].values
MD_x = (MD_x == "Yes").astype(int)
col_means = np.round(np.mean(MD_x, axis=0), 2)
print(col_means)


# In[21]:


from sklearn.decomposition import PCA
MD_pca = PCA()
MD_pca.fit(MD_x)
explained_variance_ratio = MD_pca.explained_variance_ratio_
singular_values = MD_pca.singular_values_
components = MD_pca.components_
print("Proportion of Variance Explained:")
print(explained_variance_ratio)
print("\nSingular Values:")
print(singular_values)
print("\nPrincipal Components:")
print(components)


# In[24]:


summary_string = str(MD_pca)
summary_string = summary_string.replace("\n", "\n").replace("e+", "e").replace("e-", "e")
digits = 1
print(summary_string)


# In[25]:


print("Standard deviations:")
print(np.round(np.sqrt(MD_pca.explained_variance_), 1))

print("\nProportion of Variance Explained:")
print(np.round(MD_pca.explained_variance_ratio_, 1))

print("\nRotation Matrix:")
print(np.round(MD_pca.components_, 1))


# In[26]:


import matplotlib.pyplot as plt
transformed_data = MD_pca.transform(MD_x)
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], color='grey')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection")
plt.show()


# In[30]:


from sklearn.cluster import KMeans
np.random.seed(1234)
cluster_range = range(2, 9)
best_kmeans = None
best_score = None
for n_clusters in cluster_range:
     kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=1234)
     kmeans.fit(MD_x)
     score = -kmeans.inertia_
     if best_score is None or score > best_score:
        best_score = score
        best_kmeans = kmeans
labels = best_kmeans.labels_
print(labels)


# In[35]:


scores = []
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=1234)
    kmeans.fit(MD_x)
    score = -kmeans.inertia_
    scores.append(score)
plt.bar(cluster_range, scores)
plt.xlabel("Number of segments")
plt.ylabel("Sum of within cluster distance")
plt.title("K-means Clustering")
plt.show()


# In[43]:


from sklearn.utils import resample
n_boot = 100
boot_results = []
for n_clusters in cluster_range:
     bootstrap_scores = []
     for _ in range(n_boot):
        resampled_data = resample(MD_x, random_state=1234)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=1234)
        kmeans.fit(resampled_data)
        score = -kmeans.inertia_
        bootstrap_scores.append(score)
     boot_results.append(bootstrap_scores)
boot_results = np.array(boot_results)
print(boot_results)


# In[45]:


from sklearn.metrics import adjusted_rand_score
bootstrap_results = []
for n_clusters in cluster_range:
    bootstrap_scores = []
    for _ in range(n_boot):
        resampled_data = resample(MD_x, random_state=1234)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=1234)
        kmeans.fit(resampled_data)
        true_labels = np.argmax(resampled_data, axis=1)
        predicted_labels = kmeans.labels_
        adjusted_rand_index = adjusted_rand_score(true_labels, predicted_labels)
        bootstrap_scores.append(adjusted_rand_index)
    bootstrap_results.append(bootstrap_scores)
data = bootstrap_results
plt.boxplot(data, labels=cluster_range)
plt.xlabel("Number of segments")
plt.ylabel("Adjusted Rand index")
plt.title("Bootstrapped Results")
plt.show()


# In[51]:


np.random.seed(1234)
n_clusters = 8 
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=1234)
kmeans.fit(MD_x)
desired_cluster = 4
if desired_cluster < 1 or desired_cluster > n_clusters:
    print(f"Invalid cluster number. Please choose a cluster between 1 and {n_clusters}.")
else:
    cluster_indices = np.where(kmeans.labels_ == desired_cluster - 1)[0]
    if len(cluster_indices) == 0:
        print(f"No values found for cluster {desired_cluster}.")
    else:
        values_of_cluster = MD_x[cluster_indices, desired_cluster - 1]
        plt.hist(values_of_cluster, bins=10, range=(0, 1))
        plt.xlabel("Values")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of Cluster {desired_cluster}")
        plt.xlim(0, 1)
        plt.show()


# In[60]:


from sklearn.metrics import silhouette_samples
np.random.seed(1234)
n_clusters = 8  
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=1234)
kmeans.fit(MD_x)
desired_cluster = 4
if desired_cluster < 1 or desired_cluster > n_clusters:
    print(f"Invalid cluster number. Please choose a cluster between 1 and {n_clusters}.")
else:
    cluster_labels = kmeans.labels_
    values_of_cluster = MD_x[cluster_labels == desired_cluster - 1, :]
    silhouette_vals = silhouette_samples(MD_x, cluster_labels)
    segment_nums = np.arange(1, len(silhouette_vals) + 1)
    plt.plot(segment_nums, silhouette_vals)
    plt.ylim(0, 1)
    plt.xlabel("Segment number")
    plt.ylabel("Segment stability")
    plt.title("Segment Level Stability")
    plt.show()


# In[62]:


conda install -m rpy2


# In[2]:


pip install rpy2


# In[5]:


import rpy2.situation
r_home = rpy2.situation.get_r_home()
print(f"R home directory: {r_home}")


# In[17]:


import pandas as pd
data = pd.read_csv(r"C:\Users\sonas\Downloads\mcdonalds.csv")
MD_x = data.iloc[:, 0:11].values
MD_x = (MD_x == "Yes").astype(int)
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix


# In[18]:


import numpy as np
from sklearn.mixture import GaussianMixture
np.random.seed(1234)
n_clusters = np.arange(2, 9) 
MD_m28 = []

for n in n_clusters:
    gmm = GaussianMixture(n_components=n, n_init=10, random_state=1234)
    gmm.fit(MD_x)
    MD_m28.append(gmm)
for i, model in enumerate(MD_m28):
    print(f"Number of clusters: {n_clusters[i]}")
    print(model)
    print()


# In[22]:


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
n_clusters = 8  
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=1234)
kmeans.fit(MD_x)
desired_cluster_kmeans = 4
values_of_cluster_kmeans = MD_x[kmeans.labels_ == desired_cluster_kmeans - 1]
desired_cluster_model = 4
MD_m4_model = MD_m28[desired_cluster_model - 1] if desired_cluster_model <= len(MD_m28) else None
if MD_m4_model is not None:
    predicted_labels_kmeans = kmeans.labels_
    predicted_labels_model = MD_m4_model.predict(MD_x)
    contingency_table = confusion_matrix(predicted_labels_kmeans, predicted_labels_model)
    print(contingency_table)
else:
    print(f"No model found for cluster {desired_cluster_model}.")


# In[25]:



data.fillna(data.median())


# In[28]:


del data['Like']


# In[29]:


data.head()


# In[35]:


import matplotlib.pyplot as plt
aic_values = []
bic_values = []
icl_values = []
for model in MD_m28:
    aic_values.append(model.aic)
    bic_values.append(model.bic)
    icl_values.append(model.lower_bound_)
plt.plot(aic_values, label="AIC")
plt.plot(bic_values, label="BIC")
plt.plot(icl_values, label="ICL")
plt.xlabel("Number of clusters")
plt.ylabel("Value of information criteria")
plt.title("Information Criteria for Model-Based Clustering")
plt.legend()
plt.show()


# In[37]:


from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
n_clusters = 8  
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=1234)
kmeans.fit(MD_x)
cluster_assignments_kmeans = kmeans.labels_
gmm = GaussianMixture(n_components=n_clusters, n_init=10, random_state=1234)
gmm.fit(MD_x)
cluster_assignments_m4a = gmm.predict(MD_x)
contingency_table = confusion_matrix(cluster_assignments_kmeans, cluster_assignments_m4a)
print(contingency_table)


# In[44]:


MD_m4 = []
from sklearn.mixture import GaussianMixture
n_clusters = 8
gmm = GaussianMixture(n_components=n_clusters, n_init=10, random_state=1234)
gmm.fit(MD_x)
log_likelihood_m4a = gmm.score_samples(MD_x).sum()
log_likelihood_m4 = gmm.score(MD_x)
print(f"Log Likelihood (MD.m4a): {log_likelihood_m4a}")
print(f"Log Likelihood (MD.m4): {log_likelihood_m4}")


# In[3]:


import pandas as pd
data1 = pd.read_csv(r"C:\Users\sonas\Downloads\mcdonalds.csv")


# In[4]:


data1.head()


# In[14]:


#Using Mixtures of Regression Models


# In[16]:


import numpy as np
from collections import Counter
likes = data1["Like"]
like_counts = dict(Counter(likes))
reversed_table = {key: value for key, value in reversed(like_counts.items())}
for key, value in reversed_table.items():
    print(f"{key}: {value}")


# In[18]:


import pandas as pd
data1["Like.n"] = 6 - pd.to_numeric(data1["Like"], errors='coerce')
like_n_counts = data1["Like.n"].value_counts()
print(like_n_counts)


# In[24]:


import pandas as pd
from patsy import dmatrices
f = " + ".join(data1.columns[1:12])
f = "Like.n ~ " + f
formula = (data1.apply(dmatrices(f, data=data1, return_type='dataframe').design_info))
print(formula)


# In[32]:


data1['yummy'] = data1['yummy'].replace({'Yes': 1, 'No': 0})
data1['convenient'] = data1['convenient'].replace({'Yes': 1, 'No': 0})


# In[33]:


data1.head()


# In[34]:


from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import check_random_state
random_state = check_random_state(1234)
categorical_columns = ['yummy', 'convenient']  
transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), categorical_columns)],
                                remainder='passthrough')
data_encoded = transformer.fit_transform(data1)
MD_reg2 = GaussianMixture(n_components=2, n_init=10, random_state=random_state)
MD_reg2.fit(data_encoded)
print(MD_reg2)


# In[36]:


from sklearn.impute import SimpleImputer

random_state = check_random_state(1234)
categorical_columns = ['yummy', 'convenient'] 
numeric_columns = ['Age', 'yummy']  
preprocessor = ColumnTransformer([
    ('categorical_encoder', OneHotEncoder(), categorical_columns),
    ('numeric_imputer', SimpleImputer(strategy='mean'), numeric_columns)
])

data_preprocessed = preprocessor.fit_transform(data1)
MD_reg2 = GaussianMixture(n_components=2, n_init=10, random_state=random_state)
MD_reg2.fit(data_preprocessed)
print(MD_reg2)


# In[38]:


data1['spicy'] = data1['spicy'].replace({'Yes': 1, 'No': 0})
data1['fattening'] = data1['fattening'].replace({'Yes': 1, 'No': 0})
data1['greasy'] = data1['greasy'].replace({'Yes': 1, 'No': 0})
data1['fast'] = data1['fast'].replace({'Yes': 1, 'No': 0})
data1['cheap'] = data1['cheap'].replace({'Yes': 1, 'No': 0})
data1['tasty'] = data1['tasty'].replace({'Yes': 1, 'No': 0})
data1['expensive'] = data1['expensive'].replace({'Yes': 1, 'No': 0})
data1['healthy'] = data1['healthy'].replace({'Yes': 1, 'No': 0})
data1['disgusting'] = data1['disgusting'].replace({'Yes': 1, 'No': 0})
data1['Gender'] = data1['Gender'].replace({'Male': 1, 'Female': 0})


# In[40]:


del data1['Like']


# In[41]:


del data1['VisitFrequency']


# In[42]:


data1.head()


# In[45]:


data1 = data1.fillna(0)


# In[46]:


MD_ref2 = MD_reg2.fit(data1)
log_likelihood = MD_ref2.score(data1)
print("Log-Likelihood:", log_likelihood)
print("Number of components:", MD_ref2.n_components)
print("Means:")
print(MD_ref2.means_)
print("Covariances:")
print(MD_ref2.covariances_)


# In[49]:


pip install matplotlib


# In[50]:





# In[ ]:




