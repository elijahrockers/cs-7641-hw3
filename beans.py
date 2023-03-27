import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection

# Load dataset
df = pd.read_csv('beans.csv')

# Sample dataset (5% subset) 
pct = .5
df = df.sample(int(len(df)*pct))

print("N: ", len(df))

# Preprocessing
X = df.drop('Class', axis=1)
attrs = X.columns.tolist()
stdscaler = StandardScaler()
X[attrs] = stdscaler.fit_transform(X[attrs])
le = LabelEncoder()
df['Class'] = le.fit_transform(df.Class.values)
mapping = dict(zip(le.classes_, range(len(le.classes_))))
y = df['Class']
df = X

# Split data into training and testing 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# X_train = X_train[['ShapeFactor4', 'ShapeFactor1', 'MinorAxisLength', 'Solidity', 'ShapeFactor2']]
# X_test = X_test[['ShapeFactor4', 'ShapeFactor1', 'MinorAxisLength', 'Solidity', 'ShapeFactor2']]

# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)
# importances = pd.DataFrame(data={
#     'Attribute': X_train.columns,
#     'Importance': model.coef_[0]
# })
# importances = importances.sort_values(by='Importance', ascending=False)

# plt.figure()
# plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
# plt.title('Feature importances obtained from coefficients', size=20)
# plt.xticks(rotation='vertical')
# plt.show()

col_lab =[(0, 'black'),
          (1, 'red'),
          (2, 'green'),
          (3, 'blue'),
          (4, 'orange'),
          (5, 'yellow'),
          (6, 'purple')]

############# KMEANS - UNREDUCED
kmeans = KMeans(n_clusters=7, max_iter=300, tol=0.0001, n_init=10)
pred = kmeans.fit_predict(df, None)
fig = plt.figure()
fig.suptitle("K Means, K=7")
labels = []
for label in range(0, 7):
    labels.append(df[pred == label])
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
for label, col in col_lab:
    ax1.scatter(labels[label]['ShapeFactor1'], labels[label]['ShapeFactor2'], marker=".", s=10, c=col)
    ax1.set(xlabel="ShapeFactor1", ylabel="ShapeFactor2")
    ax2.scatter(labels[label]['MajorAxisLength'], labels[label]['MinorAxisLength'], marker=".", s=10, c=col)
    ax2.set(xlabel="MajorAxisLength", ylabel="MinorAxisLength")
    ax3.scatter(labels[label]['ShapeFactor3'], labels[label]['ShapeFactor4'], marker=".", s=10, c=col)
    ax3.set(xlabel="ShapeFactor3", ylabel="ShapeFactor4")
    ax4.scatter(labels[label]['roundness'], labels[label]['Area'], marker=".", s=10, c=col)
    ax4.set(xlabel="roundness", ylabel="Area")
# plt.show()

############# GMM - UNREDUCED
gauss = GaussianMixture(n_components=7, n_init=10, init_params='kmeans')
pred = gauss.fit_predict(df)
fig = plt.figure()
fig.suptitle("GMM, K=7")
labels = []
for label in range(0, 7):
    labels.append(df[pred == label])
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
for label, col in col_lab:
    ax1.scatter(labels[label]['ShapeFactor1'], labels[label]['ShapeFactor2'], marker=".", s=10, c=col)
    ax1.set(xlabel="ShapeFactor1", ylabel="ShapeFactor2")
    ax2.scatter(labels[label]['MajorAxisLength'], labels[label]['MinorAxisLength'], marker=".", s=10, c=col)
    ax2.set(xlabel="MajorAxisLength", ylabel="MinorAxisLength")
    ax3.scatter(labels[label]['ShapeFactor3'], labels[label]['ShapeFactor4'], marker=".", s=10, c=col)
    ax3.set(xlabel="ShapeFactor3", ylabel="ShapeFactor4")
    ax4.scatter(labels[label]['roundness'], labels[label]['Area'], marker=".", s=10, c=col)
    ax4.set(xlabel="roundness", ylabel="Area")
# plt.show()

################## PCA
pca = PCA(2)
pca_df = pca.fit_transform(df)

kmeans = KMeans(n_clusters=7, n_init=10)
pred = kmeans.fit_predict(pca_df)
labels = []
for label in range(0, 7):
    labels.append(pca_df[pred == label])
plt.figure()
plt.title("K-Means (k=7), after PCA(2)")
for label, col in col_lab:
    plt.scatter(labels[label][:,0], labels[label][:,1], marker=".", s=10, c=col)

# plt.show()

gauss = GaussianMixture(n_components=7, n_init=10)
pred = gauss.fit_predict(pca_df)
labels = []
for label in range(0, 7):
    labels.append(pca_df[pred == label])
plt.figure()
plt.title("GMM (n=7), after PCA(2)")
for label, col in col_lab:
    plt.scatter(labels[label][:,0], labels[label][:,1], marker=".", s=10, c=col)

# plt.show()

################## ICA
ica = FastICA(n_components=7, max_iter=1000, tol=0.01, whiten='unit-variance')
ica_df = ica.fit_transform(df)

kmeans = KMeans(n_clusters=7, n_init=10)
pred = kmeans.fit_predict(ica_df)
labels = []
for label in range(0, 7):
    labels.append(ica_df[pred == label])
plt.figure()
plt.title("K-Means (k=7), after FastICA(7)")
for label, col in col_lab:
    plt.scatter(labels[label][:,0], labels[label][:,1], marker=".", s=10, c=col)

# plt.show()

gauss = GaussianMixture(n_components=7, n_init=10)
pred = gauss.fit_predict(ica_df)
labels = []
for label in range(0, 7):
    labels.append(ica_df[pred == label])
plt.figure()
plt.title("GMM (n=7), after FastICA(7)")
for label, col in col_lab:
    plt.scatter(labels[label][:,0], labels[label][:,1], marker=".", s=10, c=col)

# plt.show()

################## Gaussian Random Projections
grp = GaussianRandomProjection(n_components=3)
grp_df = grp.fit_transform(df)

kmeans = KMeans(n_clusters=7, n_init=10)
pred = kmeans.fit_predict(grp_df)
labels = []
for label in range(0, 7):
    labels.append(grp_df[pred == label])
plt.figure()
plt.title("K-Means (k=7), after Gaussian Random Projection(3)")
for label, col in col_lab:
    plt.scatter(labels[label][:,0], labels[label][:,1], marker=".", s=10, c=col)

# plt.show()

gauss = GaussianMixture(n_components=7, n_init=10)
pred = gauss.fit_predict(grp_df)
labels = []
for label in range(0, 7):
    labels.append(grp_df[pred == label])
plt.figure()
plt.title("GMM (n=7), after Gaussian Random Projection(3)")
for label, col in col_lab:
    plt.scatter(labels[label][:,0], labels[label][:,1], marker=".", s=10, c=col)

# plt.show()

################## Kernel PCA
kpca = KernelPCA(n_components=2)
kpca_df = grp.fit_transform(df)

kmeans = KMeans(n_clusters=7, n_init=10)
pred = kmeans.fit_predict(kpca_df)
labels = []
for label in range(0, 7):
    labels.append(kpca_df[pred == label])
plt.figure()
plt.title("K-Means (k=7), after Kernel PCA(2)")
for label, col in col_lab:
    plt.scatter(labels[label][:,0], labels[label][:,1], marker=".", s=10, c=col)

# plt.show()

gauss = GaussianMixture(n_components=7, n_init=10)
pred = gauss.fit_predict(kpca_df)
labels = []
for label in range(0, 7):
    labels.append(kpca_df[pred == label])
plt.figure()
plt.title("GMM (n=7), after Kernel PCA(2)")
for label, col in col_lab:
    plt.scatter(labels[label][:,0], labels[label][:,1], marker=".", s=10, c=col)

plt.show()
