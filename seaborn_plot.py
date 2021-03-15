import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


CL_k_att = np.load('embedding_whole_att.npy')
CL_k_att_logit = np.load('embedding_whole_att_logit.npy')

#CL_k_nearest = CL_k_nearest[0:3998,:]
#CL_k_nearest_logit = CL_k_nearest_logit[0:3998,:]

CL_k = np.load('embedding_whole_random.npy')
CL_k_logit = np.load('embedding_whole_random_logit.npy')

kmeans_train_death_att = KMeans(n_clusters=1, random_state=0).fit(CL_k_att[np.where(CL_k_att_logit==1)[0],:])

train_death_center_att = kmeans_train_death_att.cluster_centers_

kmeans_train_live_att = KMeans(n_clusters=1, random_state=0).fit(CL_k_att[np.where(CL_k_att_logit==0)[0],:])

train_live_center_att = kmeans_train_live_att.cluster_centers_

kmeans_train_death = KMeans(n_clusters=1, random_state=0).fit(CL_k[np.where(CL_k_logit==1)[0],:])

train_death_center = kmeans_train_death.cluster_centers_

kmeans_train_live = KMeans(n_clusters=1, random_state=0).fit(CL_k[np.where(CL_k_logit==0)[0],:])

train_live_center = kmeans_train_live.cluster_centers_



#geyser = sns.load_dataset("geyser")
#sns.kdeplot(data=geyser, x="waiting", y="duration",hue="kind",fill=True,)
#plt.show()