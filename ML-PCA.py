# This code is to apply machine learning (k-means) and PCA methods for detecting
#types of rocks.
# If you use this code, please read and cite our paper "Moving Toward Real-Time
#Rock Mass Characterization Based on Drilling Data – An Application of 
#Artificial Intelligence"
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
#PCA
from sklearn.decomposition import PCA

file.columns = [''] 
file.fillna(value=0, inplace=True)
feature_columns =  ''.split()
X_old = file[feature_columns]

x_tmp = X_old.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x_tmp)
X = pd.DataFrame(x_scaled)

### Optimum number of clusters
ks = range(1, 10)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model=KMeans(n_clusters=k , random_state=3425)
    
    # Fit model to samples
    model.fit(X)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
#namebank =[0,0.61, 0.68, 0.70, 0.73, 0.80, 0.85, 0.81,0.70]
sns.set(rc={"figure.figsize": (12, 6)})
sns.set_style("white")
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k', size=20)
plt.ylabel('inertia', size=20)
plt.xticks(ks)
plt.tick_params(direction='out', length=6, width=2, colors='k')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.grid(True, which='major', axis='y', color="white", linewidth=1, zorder=1)
#plt.text(ks[0], inertias[0], namebank[0], horizontalalignment='center', verticalalignment='center', size=20)
#plt.text(ks[1], inertias[1], namebank[1], horizontalalignment='center', verticalalignment='center', size=20)
#plt.text(ks[2], inertias[2], namebank[2], horizontalalignment='center', verticalalignment='center', size=20)
#plt.text(ks[3], inertias[3], namebank[3], horizontalalignment='center', verticalalignment='center', size=20)
#plt.text(ks[4], inertias[4], namebank[4], horizontalalignment='center', verticalalignment='center', size=20)
#plt.text(ks[5], inertias[5], namebank[5], horizontalalignment='center', verticalalignment='center', size=20)
#plt.text(ks[6], inertias[6], namebank[6], horizontalalignment='center', verticalalignment='center', size=20)
#plt.text(ks[7], inertias[7], namebank[7], horizontalalignment='center', verticalalignment='center', size=20)
#plt.text(ks[8], inertias[8], namebank[8], horizontalalignment='center', verticalalignment='center', size=20)
plt.show()

# Create scaler: scaler
scaler = StandardScaler()
# Create KMeans instance: kmeans
# Select the best number of clusters
kmeans = KMeans(n_clusters=)
# Create pipeline: pipeline
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(X)
labels_s = pipeline.predict(X)
classify = pd.DataFrame({'classification':labels_s})
classify.head()
classify.to_csv('classify.csv', index=False)
print(labels_s[2])

# Create a normalizer: normalizer
normalizer = Normalizer()
kmeans = KMeans(n_clusters=, random_state=0)
# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer, kmeans)
# Fit pipeline to the daily price movements
pipeline.fit(X)
# Calculate the cluster labels: labels
labels = pipeline.predict(X)
# Create crosstab: ct
#ct = pd.crosstab(df['labels'], df['species'])

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, pca)

# Fit the pipeline to 'samples'
pipeline.fit(X)

# Plot the explained variances
features = range(pca.n_components_)
sns.set(rc={"figure.figsize": (12, 6)})
sns.set_style("white")
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature',  size=20)
plt.ylabel('variance',  size=20)
plt.xticks(features)
plt.show()

from sklearn.preprocessing import scale
scaled_samples = scale(X)
# Select the best number of components
pca = PCA(n_components=)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples) #scaled_samples

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples) 

# Print the shape of pca_features
print(pca_features.shape)

reduced_data = PCA(n_components=).fit_transform(X)
reduced_data

 ##Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(X)
kmeans = KMeans(init='k-means++', n_clusters=, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.01     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
labels = Z

# Put the result into a color plot
Z = Z.reshape(xx.shape)

sns.set(rc={"figure.figsize": (12, 6)})
sns.set_style("white")
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=6)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w',  zorder=10)
plt.title('PCA-reduced data')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

y = np.random.uniform(0, 5, size=(7,))
print(y)
#y = [0.65969493 3.22287549 2.89750594 0.92066918 0.30156325 4.50262006 4.70370447]
y_cluster = [784, 840, 1418, 1073, 1743, 1100, 365]
# y = cluster_counts   
classify1 = pd.DataFrame({'classification':y})
classify1.to_csv('classify.csv')