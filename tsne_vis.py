from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import numpy as np


CL_k_nearest = np.load('CE_k_nearest_neg.npy')
CL_k_nearest_logit = np.load('CE_k_nearest_neg_logit.npy')

CL_k = np.load('CL_k.npy')
CL_k_logit = np.load('CL_k_logit.npy')

#CL = TSNE(n_components=2).fit_transform(CL)
#CL_k = TSNE(n_components=2).fit_transform(CL_k)
CL_k = TSNE(n_components=2).fit_transform(CL_k)
CL_k_nearest = TSNE(n_components=2).fit_transform(CL_k_nearest)
#CL_k_attribute = TSNE(n_components=2).fit_transform(CL_k_attribute)


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

plt.show()
