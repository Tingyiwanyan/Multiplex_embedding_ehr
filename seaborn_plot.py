import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import umap.umap_ as umap
import math


CL_k_feature = np.load('embedding_whole_feature.npy')
CL_k_feature_logit = np.load('embedding_whole_feature_logit.npy')

CL_k_att = np.load('embedding_whole_att.npy')
CL_k_att_logit = np.load('embedding_whole_att_logit.npy')

#CL_k_nearest = CL_k_nearest[0:3998,:]
#CL_k_nearest_logit = CL_k_nearest_logit[0:3998,:]

CL_k = np.load('embedding_whole_random.npy')
CL_k_logit = np.load('embedding_whole_random_logit.npy')

"""
kmeans_train_death_att = KMeans(n_clusters=1, random_state=0).fit(CL_k_att[np.where(CL_k_att_logit==1)[0],:])

train_death_center_att = kmeans_train_death_att.cluster_centers_

kmeans_train_live_att = KMeans(n_clusters=1, random_state=0).fit(CL_k_att[np.where(CL_k_att_logit==0)[0],:])

train_live_center_att = kmeans_train_live_att.cluster_centers_

kmeans_train_death_feature = KMeans(n_clusters=1, random_state=0).fit(CL_k_feature[np.where(CL_k_feature_logit==1)[0],:])

train_death_center_feature = kmeans_train_death_feature.cluster_centers_

kmeans_train_live_feature = KMeans(n_clusters=1, random_state=0).fit(CL_k_feature[np.where(CL_k_feature_logit==0)[0],:])

train_live_center_feature = kmeans_train_live_feature.cluster_centers_

kmeans_train_death = KMeans(n_clusters=1, random_state=0).fit(CL_k[np.where(CL_k_logit==1)[0],:])

train_death_center = kmeans_train_death.cluster_centers_

kmeans_train_live = KMeans(n_clusters=1, random_state=0).fit(CL_k[np.where(CL_k_logit==0)[0],:])

train_live_center = kmeans_train_live.cluster_centers_

pos_CL_k = CL_k[np.where(CL_k_logit==1)[0],:]
neg_CL_k = CL_k[np.where(CL_k_logit==0)[0],:]

pos_CL_k_att = CL_k_att[np.where(CL_k_att==1)[0],:]
neg_CL_k_att = CL_k_att[np.where(CL_k_att==0)[0],:]

pos_CL_k_feature = CL_k_feature[np.where(CL_k_feature==1)[0],:]
neg_CL_k_feature = CL_k_feature[np.where(CL_k_feature==0)[0],:]

center_pos_att = []
center_neg_att = []
center_pos_feature = []
center_neg_feature = []
center_pos = []
center_neg = []

for i in range(1713):
    if CL_k_att_logit[i,0] == 1:
        distance_att = np.linalg.norm(CL_k_att[i,:]-train_live_center_att)
        distance_feature = np.linalg.norm(CL_k_feature[i,:]-train_live_center_feature)
        distance = np.linalg.norm(CL_k[i,:]-train_live_center)
        center_pos_att.append(distance_att)
        center_pos_feature.append(distance_feature)
        center_pos.append(distance)
    else:
        distance_att = np.linalg.norm(CL_k_att[i,:]-train_death_center_att)
        distance_feature = np.linalg.norm(CL_k_feature[i,:]-train_death_center_feature)
        distance = np.linalg.norm(CL_k[i,:]-train_death_center)
        center_neg_att.append(distance_att)
        center_neg_feature.append(distance_feature)
        center_neg.append(distance)

"""

reducer = umap.UMAP()
CL_k = reducer.fit_transform(CL_k)
CL_k_att = reducer.fit_transform(CL_k_att)
CL_k_feature = reducer.fit_transform(CL_k_feature)

"""
kmeans_train_death_att = KMeans(n_clusters=3, random_state=0).fit(CL_k_att[np.where(CL_k_att_logit==1)[0],:])
CL_k_att_death = CL_k_att[np.where(CL_k_att_logit==1)[0],:]
label_death_att = kmeans_train_death_att.labels_

#train_death_center_att = kmeans_train_death_att.cluster_centers_

kmeans_train_live_att = KMeans(n_clusters=3, random_state=0).fit(CL_k_att[np.where(CL_k_att_logit==0)[0],:])
CL_k_att_live = CL_k_att[np.where(CL_k_att_logit==0)[0],:]
label_live_att = kmeans_train_live_att.labels_+3

whole_label_att = np.concatenate((label_death_att,label_live_att))
CL_k_att_total = np.concatenate((CL_k_att_death,CL_k_att_live),0)

#train_live_center_att = kmeans_train_live_att.cluster_centers_

kmeans_train_death_feature = KMeans(n_clusters=2, random_state=0).fit(CL_k_feature[np.where(CL_k_feature_logit==1)[0],:])

#train_death_center_feature = kmeans_train_death_feature.cluster_centers_

kmeans_train_live_feature = KMeans(n_clusters=2, random_state=0).fit(CL_k_feature[np.where(CL_k_feature_logit==0)[0],:])

#train_live_center_feature = kmeans_train_live_feature.cluster_centers_

kmeans_train_death = KMeans(n_clusters=3, random_state=0).fit(CL_k[np.where(CL_k_logit==1)[0],:])

#train_death_center = kmeans_train_death.cluster_centers_

kmeans_train_live = KMeans(n_clusters=3, random_state=0).fit(CL_k[np.where(CL_k_logit==0)[0],:])
"""
#train_live_center = kmeans_train_live.cluster_centers_


CL_k_center = np.sum(CL_k,0)/1713
CL_k_att_center = np.sum(CL_k_att,0)/1713
CL_k_feature_center = np.sum(CL_k_feature,0)/1713

train_death_center = np.array(
    [[np.median(CL_k[np.where(CL_k_logit == 1)[0], 0]), np.median(CL_k[np.where(CL_k_logit == 1)[0], 1])]])
train_live_center = np.array(
    [[np.median(CL_k[np.where(CL_k_logit == 0)[0], 0]), np.median(CL_k[np.where(CL_k_logit == 0)[0], 1])]])

train_death_center_att = np.array(
    [[np.median(CL_k_att[np.where(CL_k_logit == 1)[0], 0]), np.median(CL_k_att[np.where(CL_k_logit == 1)[0], 1])]])
train_live_center_att = np.array(
    [[np.median(CL_k_att[np.where(CL_k_logit == 0)[0], 0]), np.median(CL_k_att[np.where(CL_k_logit == 0)[0], 1])]])

train_death_center_feature = np.array(
    [[np.median(CL_k_feature[np.where(CL_k_logit == 1)[0], 0]), np.median(CL_k_feature[np.where(CL_k_logit == 1)[0], 1])]])
train_live_center_feature = np.array(
    [[np.median(CL_k_feature[np.where(CL_k_logit == 0)[0], 0]), np.median(CL_k_feature[np.where(CL_k_logit == 0)[0], 1])]])
#CL_k_pos_center = np.median(CL_k[np.where(CL_k_logit==1)[0]])

#CL_k_feature[:,0] = (CL_k_feature_center-CL_k_feature)[:,0]
CL_k_logit_text = []

for i in range(1713):
    if CL_k_logit[i][0] == 1:
        CL_k_logit_text.append('Yes')
    if CL_k_logit[i][0] == 0:
        CL_k_logit_text.append('No')


CL_k = np.concatenate((CL_k,train_death_center),0)
CL_k = np.concatenate((CL_k,train_live_center),0)
CL_k_att = np.concatenate((CL_k_att,train_death_center_att),0)
CL_k_att = np.concatenate((CL_k_att,train_live_center_att),0)
CL_k_feature = np.concatenate((CL_k_feature,train_death_center_feature),0)
CL_k_feature = np.concatenate((CL_k_feature,train_live_center_feature),0)


CL_k_logit_text.append('positive center')
CL_k_logit_text.append('negative center')
#CL_k_att = np.concatenate((CL_k_att,CL_k_att_center),0)
#CL_k_feature = np.concatenate((CL_k_feature,CL_k_feature_center),0)

CL_k_logit_text = np.array(CL_k_logit_text)




def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

for i in range(1715):
    CL_k_feature[i,:] = rotate(CL_k_feature_center,CL_k_feature[i,:],-np.pi*8/9.0)

train_death_center_feature = np.array([rotate(CL_k_feature_center,train_death_center_feature[0],-np.pi*6/7.0)])
train_live_center_feature = np.array([rotate(CL_k_feature_center,train_live_center_feature[0],-np.pi*6/7.0)])

for i in range(1715):
    CL_k_att[i,:] = rotate(CL_k_att_center,CL_k_att[i,:],-np.pi/9.0)

train_death_center_att = np.array([rotate(CL_k_att_center,train_death_center_att[0],-np.pi/9.0)])
train_live_center_att = np.array([rotate(CL_k_att_center,train_live_center_att[0],-np.pi/9.0)])

"""
for i in range(1715):
    CL_k[i,:] = rotate(CL_k_center,CL_k[i,:],np.pi/8)

train_death_center = np.array([rotate(CL_k_center,train_death_center[0],np.pi/8)])
train_live_center = np.array([rotate(CL_k_center,train_live_center[0],np.pi/8)])
"""

df = pd.DataFrame({"UMAP-1":CL_k[:,0], "UMAP-2":CL_k[:,1], "Mortality":CL_k_logit_text})
#sns.kdeplot(data=df, x="waiting", y="duration",hue="kind",)#fill=True,)
k = sns.jointplot(data=df, x='UMAP-1', y='UMAP-2',hue="Mortality",marker='o',color='r').plot_joint(sns.kdeplot,levels=7)
k.ax_joint.plot(train_death_center[0][0], train_death_center[0][1], marker='o',color='g',ms=10)
k.ax_joint.plot(train_live_center[0][0], train_live_center[0][1], marker='o',color='r',ms=10)
plt.show()
"""
df_att = pd.DataFrame({"UMAP-1":CL_k_att[:,0], "UMAP-2":CL_k_att[:,1], "Mortality":CL_k_logit_text})
k=sns.jointplot(data=df_att, x="UMAP-1", y="UMAP-2",hue="Mortality",marker='o').plot_joint(sns.kdeplot,levels=7)
k.ax_joint.plot(train_death_center_att[0][0], train_death_center_att[0][1], marker='o',color='g',ms=10)
k.ax_joint.plot(train_live_center_att[0][0], train_live_center_att[0][1], marker='o',color='r',ms=10)
plt.show()

df_feature = pd.DataFrame({"UMAP-1":CL_k_feature[:,0], "UMAP-2":CL_k_feature[:,1], "Mortality":CL_k_logit_text})
k=sns.jointplot(data=df_feature, x="UMAP-1", y="UMAP-2",hue="Mortality",marker='o').plot_joint(sns.kdeplot,levels=7)
k.ax_joint.plot(train_death_center_feature[0][0], train_death_center_feature[0][1], marker='o',color='g',ms=10)
k.ax_joint.plot(train_live_center_feature[0][0], train_live_center_feature[0][1], marker='o',color='r',ms=10)
plt.show()
"""