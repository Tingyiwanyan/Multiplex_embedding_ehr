import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
#people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
people = ('ALBUMIN', 'ALKPHOS', 'ALT', 'AGAP', 'AST', 'BASOPHIL_PERC',
       'BASOPHIL', 'TBILIRUBIN', 'BUN', 'CALCIUM', 'CHLORIDE', 'BICARB',
       'CREATININE', 'EOSINO_PERC', 'EOSINO', 'GLUCOSE', 'HCT', 'HGB',
       'LYMPHO_PERC', 'LYMPHO', 'MCHC', 'MCV', 'MPV', 'MONO_PERC', 'MONO',
       'NEUTRO_PERC', 'NEUTRO', 'PLT', 'POTASSIUM', 'PROTEIN', 'RBCCNT',
       'RDW', 'SODIUM', 'WBC')
y_pos = np.arange(len(people))
#performance = 3 + 10 * np.random.rand(len(people))
performance = [ 4.17137097e-01,  3.89400372e+01,  1.41389434e+01,
         7.86462020e+00,  1.74848282e+01,  0.00000000e+00,
         0.00000000e+00, -1.17246283e-01,  1.98753394e+01,
         4.46764785e+00,  7.02929211e+01,  1.49913978e+01,
         2.71543714e-01,  0.00000000e+00,  0.00000000e+00,
         1.57688310e+02,  1.29772513e+01,  3.40329301e+00,
         2.02375438e+00,  6.25000000e-03,  9.70003360e+00,
         2.91104839e+01,  2.85288978e+00,  1.29576613e+00,
        -5.22177419e-02,  2.22622984e+01,  1.77434036e+00,
         6.73407901e+01,  2.48578629e+00,  1.64462366e+00,
         8.66814516e-01,  4.56001344e+00,  9.56114247e+01,
         3.70387251e+00]
error = np.random.rand(len(people))

ax.barh(y_pos, performance, align='center',color='orange')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Scaled Vlues')
ax.set_title('Cluter 4')

plt.show()