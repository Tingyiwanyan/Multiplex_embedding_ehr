import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve

probs = np.loadtxt('out_logit_fl.txt')
y_test = np.loadtxt('real_logit.txt')

ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

frac_of_pos, mean_pred_value = calibration_curve(y_test, probs, n_bins=10)

ax1.plot(mean_pred_value, frac_of_pos, "s-", label='calibration')
ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title(f'Calibration plot ()')

ax2.hist(probs, range=(0, 1), bins=10, label='calibration', histtype="step", lw=2)
ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")

