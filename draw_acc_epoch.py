import numpy as np
import matplotlib.pyplot as plt










plt.xlabel("Epoch")
plt.ylabel("AUPRC")
plt.title("Mortality Prediction", fontsize=14)
#plt.xlim(0.0, 1.0)
#plt.ylim(0.0, 1.0)
#x = [0.0, 1.0]
#plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random')

plt.plot(recall_ce_rnn_death_24, precision_ce_rnn_death, color='green', linestyle='dashed',linewidth=2, label='RNN+CE(AUC=0.823)')


plt.plot(recall_ce_death,precision_ce_death,color='blue',linestyle='dashed',linewidth=2,label='RETAIN+CE(AUC=0.837)')

#plt.plot(fp_rate_hl_retain,tp_rate_hl_retain,color='orange',label='RETAIN+HL')

plt.plot(precision_cl_rnn_death,recall_cl_rnn_death_24,color='violet',linewidth=1.5,label='RNN+CL(AUC=0.901)')
plt.plot(recall_cl_retain_death, precition_cl_retain, color='red', linewidth=1.5, label='RETAIN+CL(AUC=0.887)')


plt.legend(loc='lower right')
plt.show()