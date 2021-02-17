import numpy as np
import matplotlib.pyplot as plt


Epoch = [0,1,2,3,4]
auprc_ce =[0.374,0.505,0.521,0.533,0.549]
auprc_k_ce=[0.389,0.456,0.531,0.551,0.577]
auprc_knn_ce=[0.384,0.513,0.553,0.568,0.586]
auprc_fl=[0.382,0.526,0.570,0.600,0.616]
auprc_k_fl=[0.479,0.631,0.645,0.654,0.656]
auprc_knn_fl=[0.473,0.643,0.658,0.659,0.663]







plt.xlabel("Epoch")
plt.ylabel("AUPRC")
plt.title("Mortality Prediction", fontsize=14)
#plt.xlim(0, 4)
plt.ylim(0.0, 1.0)
#x = [0.0, 1.0]
#plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random')

plt.plot(Epoch, auprc_ce, color='green', linestyle='dashed',linewidth=1, label='CE')


plt.plot(Epoch,auprc_k_ce,color='blue',linestyle='dashed',linewidth=1,label='CE(k random)')

#plt.plot(fp_rate_hl_retain,tp_rate_hl_retain,color='orange',label='RETAIN+HL')

plt.plot(Epoch,auprc_knn_ce,color='violet',linestyle='dashed', linewidth=1,label='CE(knn)')
plt.plot(Epoch, auprc_fl, color='red', linewidth=1, label='FL')
plt.plot(Epoch, auprc_k_fl, color='orange', linewidth=1, label='FL(k random)')
plt.plot(Epoch, auprc_knn_fl, color='purple', linewidth=1, label='FL(knn)')


plt.legend(loc='lower right')
plt.show()