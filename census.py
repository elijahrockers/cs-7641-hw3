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
df = pd.read_csv('census.csv')

# Sample dataset
pct = .5
df = df.sample(int(len(df)*pct))
print("Total N: ", len(df))

# Preprocessing
categorical_atts = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
continuous_atts = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

df.loc[df['native-country'] != 'United-States', 'native-country'] = "Other-Country"
df = df.drop('fnlwgt', axis=1)
df = df.drop('education', axis=1)

# One hot enconding
for att in categorical_atts:
    onehot = pd.get_dummies(df[att])
    if '?' in onehot.columns:
        onehot = onehot.drop('?', axis=1)
    df = df.drop(att, axis=1)
    df = df.join(onehot)

# Label binarization
lb = LabelBinarizer()
binary = pd.DataFrame(lb.fit_transform(df['class']), index=df['class'].index, columns=['class'])
df['class'] = binary

y = df['class']
X = df.drop('class', axis=1)
df = X

# Split data into training and testing 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Standardization
scale = StandardScaler()
df_cont = df[continuous_atts]
scale.fit(df_cont)
scaled_df = scale.transform(df_cont)
scaled_df = pd.DataFrame(scaled_df, index=df_cont.index, columns=df_cont.columns)
df[continuous_atts] = scaled_df


# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)
# importances = pd.DataFrame(data={
#     'Attribute': X_train.columns,
#     'Importance': model.coef_[0]
# })
# importances = importances.sort_values(by='Importance', ascending=False)
# 
# plt.figure()
# plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
# plt.title('Feature importances obtained from coefficients', size=20)
# plt.xticks(rotation='vertical')
# plt.show()

#X_train = X_train[['capital-gain', 'Married-civ-spouse', 'Cambodia', 'Priv-house-serv', 'Columbia', 'Preschool']]
#X_test = X_test[['capital-gain', 'Married-civ-spouse', 'Cambodia', 'Priv-house-serv', 'Columbia', 'Preschool']]


################################ UNREDUCED
kmeans = KMeans(n_clusters=2, max_iter=300, tol=0.0001, n_init=10)
pred = kmeans.fit_predict(df, None)
centroids = kmeans.cluster_centers_
centdf = pd.DataFrame(centroids, columns=df.columns)

label0 = df[pred == 0]
label1 = df[pred == 1]

fig = plt.figure()
fig.suptitle("KMeans, K=2")
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.scatter(label0['age'], label0['capital-gain'], c='red', marker=".", s=10)
ax1.scatter(label1['age'], label1['capital-gain'], c='blue', marker=".", s=10)
ax1.scatter(centdf['age'], centdf['capital-gain'], c='black', marker="o", s=80)
ax1.set(ylabel="Capital Gain", xlabel="Age")
ax2.scatter(label0['age'], label0['capital-loss'], c='red', marker=".", s=10)
ax2.scatter(label1['age'], label1['capital-loss'], c='blue', marker=".", s=10)
ax2.scatter(centdf['age'], centdf['capital-loss'], c='black', marker="o", s=80)
ax2.set(ylabel="Capital Loss", xlabel="Age")
ax3.scatter(label0['age'], label0['hours-per-week'], c='red', marker=".", s=10)
ax3.scatter(label1['age'], label1['hours-per-week'], c='blue', marker=".", s=10)
ax3.scatter(centdf['age'], centdf['hours-per-week'], c='black', marker="o", s=80)
ax3.set(ylabel="Hours Per Week", xlabel="Age")
ax4.scatter(label0['age'], label0['education-num'], c='red', marker=".", s=10)
ax4.scatter(label1['age'], label1['education-num'], c='blue', marker=".", s=10)
ax4.scatter(centdf['age'], centdf['education-num'], c='black', marker="o", s=80)
ax4.set(ylabel="Education Number", xlabel="Age")
#plt.show()

gauss = GaussianMixture(n_components=2, n_init=10, init_params='kmeans')
pred = gauss.fit_predict(df)
centdf = pd.DataFrame(gauss.means_, columns=df.columns)

label0 = df[pred == 0]
label1 = df[pred == 1]

fig = plt.figure()
fig.suptitle("GMM, N=2")
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.scatter(label0['age'], label0['capital-gain'], c='red', marker=".", s=10)
ax1.scatter(label1['age'], label1['capital-gain'], c='blue', marker=".", s=10)
ax1.scatter(centdf['age'], centdf['capital-gain'], c='orange', marker="D", s=80)
ax1.set(ylabel="Capital Gain", xlabel="Age")
ax2.scatter(label0['age'], label0['capital-loss'], c='red', marker=".", s=10)
ax2.scatter(label1['age'], label1['capital-loss'], c='blue', marker=".", s=10)
ax2.scatter(centdf['age'], centdf['capital-loss'], c='orange', marker="D", s=80)
ax2.set(ylabel="Capital Loss", xlabel="Age")
ax3.scatter(label0['age'], label0['hours-per-week'], c='red', marker=".", s=10)
ax3.scatter(label1['age'], label1['hours-per-week'], c='blue', marker=".", s=10)
ax3.scatter(centdf['age'], centdf['hours-per-week'], c='orange', marker="D", s=80)
ax3.set(ylabel="Hours Per Week", xlabel="Age")
ax4.scatter(label0['age'], label0['education-num'], c='red', marker=".", s=10)
ax4.scatter(label1['age'], label1['education-num'], c='blue', marker=".", s=10)
ax4.scatter(centdf['age'], centdf['education-num'], c='orange', marker="D", s=80)
ax4.set(ylabel="Education Number", xlabel="Age")
#plt.show()

######################## PCA
pca = PCA(2)
pca_df = pca.fit_transform(df)
kmeans = KMeans(n_clusters=2, max_iter=300, tol=0.0001, n_init=10)
label = kmeans.fit_predict(pca_df)
label0 = pca_df[label == 0]
label1 = pca_df[label == 1]
fig = plt.figure()
plt.title("KMeans after PCA(2), k=2")
plt.scatter(label0[:,0], label0[:,1], c="red", marker=".", label="Cluster 0", s=10)
plt.scatter(label1[:,0], label1[:,1], c="blue", marker=".", label="Cluster 1", s=10)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1], s=80, c="black", marker="o")
#plt.show()

pred = gauss.fit_predict(pca_df)
label0 = pca_df[pred == 0]
label1 = pca_df[pred == 1]
fig = plt.figure()
plt.title("GMM after PCA(2)")
plt.scatter(label0[:,0], label0[:,1], c="red", marker=".", label="Cluster 0", s=10)
plt.scatter(label1[:,0], label1[:,1], c="blue", marker=".", label="Cluster 1", s=10)
centdf = gauss.means_
plt.scatter(centdf[:,0], centdf[:,1], s=80, c="orange", marker="D")
#plt.show()

################## ICA
ica = FastICA(n_components=2, whiten='unit-variance')
ica_df = ica.fit_transform(df)
kmeans = KMeans(n_clusters=2, max_iter=300, tol=0.0001, n_init=10)
label = kmeans.fit_predict(ica_df)
label0 = ica_df[label == 0]
label1 = ica_df[label == 1]
fig = plt.figure()
plt.title("KMeans after ICA(2), k=2")
plt.scatter(label0[:,0], label0[:,1], c="red", marker=".", label="Cluster 0", s=10)
plt.scatter(label1[:,0], label1[:,1], c="blue", marker=".", label="Cluster 1", s=10)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1], s=80, c="black", marker="o")
#plt.show()

pred = gauss.fit_predict(ica_df)
label0 = ica_df[pred == 0]
label1 = ica_df[pred == 1]
fig = plt.figure()
plt.title("GMM after ICA(2)")
plt.scatter(label0[:,0], label0[:,1], c="red", marker=".", label="Cluster 0", s=10)
plt.scatter(label1[:,0], label1[:,1], c="blue", marker=".", label="Cluster 1", s=10)
centdf = gauss.means_
plt.scatter(centdf[:,0], centdf[:,1], s=80, c="orange", marker="D")
#plt.show()

##################### Gaussian Random Projection
grp = GaussianRandomProjection(n_components=2)
grp_df = grp.fit_transform(df)
kmeans = KMeans(n_clusters=2, max_iter=300, tol=0.0001, n_init=10)
label = kmeans.fit_predict(grp_df)
label0 = grp_df[label == 0]
label1 = grp_df[label == 1]
fig = plt.figure()
plt.title("KMeans after GRP(2), k=2")
plt.scatter(label0[:,0], label0[:,1], c="red", marker=".", label="Cluster 0", s=10)
plt.scatter(label1[:,0], label1[:,1], c="blue", marker=".", label="Cluster 1", s=10)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1], s=80, c="black", marker="o")
#plt.show()

pred = gauss.fit_predict(grp_df)
label0 = grp_df[pred == 0]
label1 = grp_df[pred == 1]
fig = plt.figure()
plt.title("GMM after GRP(2)")
plt.scatter(label0[:,0], label0[:,1], c="red", marker=".", label="Cluster 0", s=10)
plt.scatter(label1[:,0], label1[:,1], c="blue", marker=".", label="Cluster 1", s=10)
centdf = gauss.means_
plt.scatter(centdf[:,0], centdf[:,1], s=80, c="orange", marker="D")
#plt.show()

######################## Kernel PCA
kpca = KernelPCA(n_components=2)
kpca_df = kpca.fit_transform(df)
kmeans = KMeans(n_clusters=2, max_iter=300, tol=0.0001, n_init=10)
label = kmeans.fit_predict(kpca_df)
label0 = kpca_df[label == 0]
label1 = kpca_df[label == 1]
fig = plt.figure()
plt.title("KMeans after KernelPCA(2), k=2")
plt.scatter(label0[:,0], label0[:,1], c="red", marker=".", label="Cluster 0", s=10)
plt.scatter(label1[:,0], label1[:,1], c="blue", marker=".", label="Cluster 1", s=10)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1], s=80, c="black", marker="o")
#plt.show()

pred = gauss.fit_predict(kpca_df)
label0 = kpca_df[pred == 0]
label1 = kpca_df[pred == 1]
fig = plt.figure()
plt.title("GMM after KernelPCA(2), n=2")
plt.scatter(label0[:,0], label0[:,1], c="red", marker=".", label="Cluster 0", s=10)
plt.scatter(label1[:,0], label1[:,1], c="blue", marker=".", label="Cluster 1", s=10)
centdf = gauss.means_
plt.scatter(centdf[:,0], centdf[:,1], s=80, c="orange", marker="D")
plt.legend()
plt.show()

