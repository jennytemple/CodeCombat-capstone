import numpy as np
import matplotlib.pyplot as plt

num_mods = [1, 2, 3, 4, 5]
m_precision = [.72, .71, .87, .80, .78]
b_precision = [.60, .66, .84, .56, .60]

ax = plt.subplot(111, title='Precision by Model')
ax.bar(num_mods, m_precision, color='g')
ax.bar(num_mods, b_precision, color='grey')
vals = ax.get_yticks()
ax.set_yticklabels(['{:3.0f}%'.format(x * 100) for x in vals])
ax.set_xlabel("Model Number")
ax.set_ylabel("Percent Accuracy")
ax.legend("test")
plt.show()
