import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import umap.umap_ as umap


CL_k_att = np.load('embedding_whole_att.npy')
CL_k_att_logit = np.load('embedding_whole_att_logit.npy')

#CL_k_nearest = CL_k_nearest[0:3998,:]
#CL_k_nearest_logit = CL_k_nearest_logit[0:3998,:]

CL_k = np.load('embedding_whole_random.npy')
CL_k_logit = np.load('embedding_whole_random_logit.npy')

reducer = umap.UMAP()
CL_k = reducer.fit_transform(CL_k)
CL_k_att = reducer.fit_transform(CL_k_att)

kmeans_train_death_att = KMeans(n_clusters=1, random_state=0).fit(CL_k_att[np.where(CL_k_att_logit==1)[0],:])

train_death_center_att = kmeans_train_death_att.cluster_centers_

kmeans_train_live_att = KMeans(n_clusters=1, random_state=0).fit(CL_k_att[np.where(CL_k_att_logit==0)[0],:])

train_live_center_att = kmeans_train_live_att.cluster_centers_

kmeans_train_death = KMeans(n_clusters=1, random_state=0).fit(CL_k[np.where(CL_k_logit==1)[0],:])

train_death_center = kmeans_train_death.cluster_centers_

kmeans_train_live = KMeans(n_clusters=1, random_state=0).fit(CL_k[np.where(CL_k_logit==0)[0],:])

train_live_center = kmeans_train_live.cluster_centers_


df = pd.DataFrame({"axes1":CL_k[:,0], "axes2":CL_k[:,1], "Mortality Label":CL_k_logit[:,0]})

#sns.kdeplot(data=df, x="waiting", y="duration",hue="kind",)#fill=True,)
sns.jointplot(data=df, x='axes1', y='axes2',hue="Mortality Label",marker='o').plot_joint(sns.kdeplot,levels=7)
plt.show()

df_att = pd.DataFrame({"axes1":CL_k_att[:,0], "axes2":CL_k_att[:,1], "Mortality Label":CL_k_att_logit[:,0]})
sns.jointplot(data=df_att, x="axes1", y="axes2",hue="Mortality Label",marker='o').plot_joint(sns.kdeplot,levels=7)
plt.show()