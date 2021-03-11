# libraries
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use(‘seaborn-whitegrid’)
# width of the bars
barWidth = 0.3
plt.figure(figsize=(10,3))
# Choose the height of the blue bars
bars1 = [0.732, 0.845, 0.863, 0.876, 0.887]
# Choose the height of the cyan bars
bars2 = [0.803, 0.916, 0.926, 0.934, 0.935]
bars3 = [0.886, 0.931, 0.941, 0.947, 0.949]
bars4 = [0.897, 0.937, 0.944, 0.952, 0.956]
bars5 = [0.906, 0.938, 0.949, 0.953, 0.954]
# Choose the height of the error bars (bars1)
yer1 = [0.02, 0.01, 0.005, 0.01, 0.015]
yer2 = [0.015, 0.013, 0.005, 0.006, 0.01]
yer3 = [0.003, 0.01, 0.005, 0.004, 0.003]
yer4 = [0.02, 0.01, 0.005, 0.004, 0.003]
yer5 = [0.02, 0.01, 0.005, 0.004, 0.003]
# The x position of bars
r1 = np.array([0,2,4,6,8])
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
# Create blue bars
plt.bar(r1, bars1, width=barWidth, alpha=0.5, color ='C0', edgecolor='black', yerr=yer1, capsize=7, label='CE')
plt.yticks(np.arange(0, 1, 0.03))
# Create cyan bars
plt.bar(r2, bars2, width=barWidth, alpha=0.5, color = 'C1', edgecolor='black', yerr=yer2, capsize=7, label='FL')
plt.bar(r3, bars3, width=barWidth, alpha=0.5, color = 'C2', edgecolor='black', yerr=yer3, capsize=7, label='FL(random)')
plt.bar(r4, bars4, width=barWidth, alpha=0.5, color = 'C3', edgecolor='black', yerr=yer4, capsize=7, label='FL(feature)')
plt.bar(r5, bars5, width=barWidth, alpha=0.5, color = 'C4', edgecolor='black', yerr=yer5, capsize=7, label='FL(attribute)')
# general layout
plt.xticks([r + 2*barWidth for r in r1], ['399/23%', '999/23%', '1999/23%','2999/23%','3999/23%'])
plt.ylabel('AUROC')
plt.xlabel('size/positive label percentage')
plt.ylim(0.65, 1)
#plt.legend(loc = “lower right”)
plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
# Show graphic
#plt.show()
#plt.savefig(“output.pdf”, format=‘pdf’,bbox_inches = ‘tight’)
