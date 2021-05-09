from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap
import seaborn as sns
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import metrics
from scipy.spatial.distance import cdist


"""
CL_k_nearest = np.load('embedding_whole_att.npy')
CL_k_nearest_logit = np.load('embedding_whole_att_logit.npy')

CL_k_nearest = CL_k_nearest[:,0:84]
#CL_k_nearest_logit = CL_k_nearest_logit[0:3998,:]

CL_k = np.load('embedding_whole_random.npy')
CL_k_logit = np.load('embedding_whole_random_logit.npy')

CL_k = CL_k[:,0:84]
"""
#CL_k_logit = CL_k_logit[0:3998,:]

CL_k_feature = np.load('embedding_whole_feature.npy')
CL_k_feature_logit = np.load('embedding_whole_feature_logit.npy')

CL_k_att = np.load('embedding_whole_att.npy')
CL_k_att_logit = np.load('embedding_whole_att_logit.npy')

#CL_k_nearest = CL_k_nearest[0:3998,:]
#CL_k_nearest_logit = CL_k_nearest_logit[0:3998,:]

CL_k = np.load('embedding_whole_random.npy')
CL_k_logit = np.load('embedding_whole_random_logit.npy')

reducer = umap.UMAP()

#CL = TSNE(n_components=2).fit_transform(CL)
#CL_k = TSNE(n_components=2).fit_transform(CL_k)
#CL_k = TSNE(n_components=2).fit_transform(CL_k)
#CL_k_nearest = TSNE(n_components=2).fit_transform(CL_k_nearest)

#CL_k = reducer.fit_transform(CL_k)
#CL_k_nearest = reducer.fit_transform(CL_k_nearest)

#CL_k_attribute = TSNE(n_components=2).fit_transform(CL_k_attribute)

CL_k = reducer.fit_transform(CL_k)
CL_k_att = reducer.fit_transform(CL_k_att)
CL_k_feature = reducer.fit_transform(CL_k_feature)


kmeans_train_death_att = KMeans(n_clusters=3, random_state=0).fit(CL_k_att[np.where(CL_k_att_logit==1)[0],:])
CL_k_att_death = CL_k_att[np.where(CL_k_att_logit==1)[0],:]
label_death_att = kmeans_train_death_att.labels_

train_death_center_att = kmeans_train_death_att.cluster_centers_

kmeans_train_live_att = KMeans(n_clusters=3, random_state=0).fit(CL_k_att[np.where(CL_k_att_logit==0)[0],:])
CL_k_att_live = CL_k_att[np.where(CL_k_att_logit==0)[0],:]
label_live_att = kmeans_train_live_att.labels_+3

whole_label_att = np.concatenate((label_death_att,label_live_att))
CL_k_att_total = np.concatenate((CL_k_att_death,CL_k_att_live),0)

train_live_center_att = kmeans_train_live_att.cluster_centers_

kmeans_train_death_feature = KMeans(n_clusters=2, random_state=0).fit(CL_k_feature[np.where(CL_k_feature_logit==1)[0],:])

#train_death_center_feature = kmeans_train_death_feature.cluster_centers_

kmeans_train_live_feature = KMeans(n_clusters=2, random_state=0).fit(CL_k_feature[np.where(CL_k_feature_logit==0)[0],:])

#train_live_center_feature = kmeans_train_live_feature.cluster_centers_

kmeans_train_death = KMeans(n_clusters=3, random_state=0).fit(CL_k[np.where(CL_k_logit==1)[0],:])
CL_k_death = CL_k[np.where(CL_k_att_logit==1)[0],:]
label_death = kmeans_train_death.labels_

train_death_center = kmeans_train_death.cluster_centers_

kmeans_train_live = KMeans(n_clusters=3, random_state=0).fit(CL_k[np.where(CL_k_logit==0)[0],:])
CL_k_live = CL_k[np.where(CL_k_att_logit==0)[0],:]
label_live = kmeans_train_live.labels_+3

whole_label = np.concatenate((label_death,label_live))
CL_k_total = np.concatenate((CL_k_death,CL_k_live),0)

train_live_center = kmeans_train_live.cluster_centers_

"""
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)
X = CL_k[np.where(CL_k_logit==1)[0],:]
for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(CL_k[np.where(CL_k_logit==1)[0],:])#KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(CL_k[np.where(CL_k_logit==1)[0],:])

    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / X.shape[0])
    inertias.append(kmeanModel.inertia_)

    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / X.shape[0]
    mapping2[k] = kmeanModel.inertia_
"""

#fig, axs = plt.subplots(2,2)
#fig.suptitle('Icu Prediction')
"""
for i in range(CL_logit.shape[0]):
    if CL_logit[i,0] == 0:
        plt.plot(CL[i][0],CL[i][1],'.',color='red',markersize=6)
    if CL_logit[i,0] == 1:
        plt.plot(CL[i][0],CL[i][1],'.',color='blue',markersize=10)
    #axs[4, 0].set_title('A')

plt.show()
"""

"""
for i in range(CL_k_logit.shape[0]):
    if CL_k_logit[i,0] == 0:
        plt.plot(CL_k[i][0],CL_k[i][1],'.',color='red',markersize=3)
    if CL_k_logit[i,0] == 1:
        plt.plot(CL_k[i][0],CL_k[i][1],'.',color='blue',markersize=6)
    #axs[4, 1].set_title('B')
plt.show()


for i in range(CL_k_nearest_logit.shape[0]):
    if CL_k_nearest_logit[i,0] == 0:
        plt.plot(CL_k_nearest[i][0],CL_k_nearest[i][1],'.',color='red',markersize=3)
    if CL_k_nearest_logit[i,0] == 1:
        plt.plot(CL_k_nearest[i][0],CL_k_nearest[i][1],'.',color='blue',markersize=6)
    #plt.set_title('C')
"""

for i in range(CL_k_att_death.shape[0]):
    if label_death_att[i] == 0:
        plt.plot(CL_k_att_death[i][0],CL_k_att_death[i][1],'o',color='red',markersize=6)
    if label_death_att[i] == 1:
        plt.plot(CL_k_att_death[i][0],CL_k_att_death[i][1],'o',color='orange',markersize=6)
    if label_death_att[i] == 2:
        plt.plot(CL_k_att_death[i][0],CL_k_att_death[i][1],'o',color='blue',markersize=6)
    if label_death_att[i] == 3:
        plt.plot(CL_k_att_death[i][0],CL_k_att_death[i][1],'o',color='purple',markersize=6)
    if label_death_att[i] == 4:
        plt.plot(CL_k_att_death[i][0],CL_k_att_death[i][1],'o',color='cyan',markersize=6)
    if label_death_att[i] == 5:
        plt.plot(CL_k_att_death[i][0],CL_k_att_death[i][1],'o',color='green',markersize=6)

plt.show()


for i in range(CL_k_death.shape[0]):
    if label_death[i] == 0:
        plt.plot(CL_k_death[i][0],CL_k_death[i][1],'o',color='blue',markersize=6)
    if label_death[i] == 1:
        plt.plot(CL_k_death[i][0],CL_k_death[i][1],'o',color='red',markersize=6)
    if label_death[i] == 2:
        plt.plot(CL_k_death[i][0],CL_k_death[i][1],'o',color='orange',markersize=6)
    if label_death[i] == 3:
        plt.plot(CL_k_death[i][0],CL_k_death[i][1],'o',color='green',markersize=6)
    if label_death[i] == 4:
        plt.plot(CL_k_death[i][0],CL_k_death[i][1],'o',color='cyan',markersize=6)
    if label_death[i] == 5:
        plt.plot(CL_k_death[i][0],CL_k_death[i][1],'o',color='purple',markersize=6)

plt.show()
