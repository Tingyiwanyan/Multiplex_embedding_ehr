from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

CE = np.load('CE.npy')
CE_logit = np.load('CE_logit.npy')
CL_k = np.load('CL_k.npy')
CL_k_logit = np.load('CL_k_logit.npy')
CL_k_nearest = np.load('CL_k_nearset.npy')
CL_k_nearest_logit = np.load('CL_k_nearset_logit.npy')
CL_k_attribute = np.load('CL_k_attribute.npy')
CL_k_attribute_logit = np.load('CL_k_attribute_logit.npy')

CE = TSNE(n_components=2).fit_transform(CE)
CL_k = TSNE(n_components=2).fit_transform(CL_k)
CL_k_nearest = TSNE(n_components=2).fit_transform(CL_k_nearest)
CL_k_attribute = TSNE(n_components=2).fit_transform(CL_k_attribute)


#fig, axs = plt.subplots(2,2)
#fig.suptitle('Icu Prediction')

for i in range(CE_logit.shape[0]):
    if CE_logit[i,0] == 0:
        plt.plot(CE[i][0],CE[i][1],'.',color='red',markersize=6)
    if CE_logit[i,0] == 1:
        plt.plot(CE[i][0],CE[i][1],'.',color='blue',markersize=10)
    #axs[4, 0].set_title('A')

plt.show()


for i in range(CL_k_logit.shape[0]):
    if CL_k_logit[i,0] == 0:
        plt.plot(CL_k[i][0],CL_k[i][1],'.',color='red',markersize=6)
    if CL_k_logit[i,0] == 1:
        plt.plot(CL_k[i][0],CL_k[i][1],'.',color='blue',markersize=10)
    #axs[4, 1].set_title('B')
plt.show()

for i in range(CL_k_nearest_logit.shape[0]):
    if CL_k_nearest_logit[i,0] == 0:
        plt.plot(CL_k_nearest[i][0],CL_k_nearest[i][1],'.',color='red',markersize=6)
    if CL_k_nearest_logit[i,0] == 1:
        plt.plot(CL_k_nearest[i][0],CL_k_nearest[i][1],'.',color='blue',markersize=10)
    #plt.set_title('C')

plt.show()
