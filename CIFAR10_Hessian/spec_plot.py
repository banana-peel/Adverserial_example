import numpy as np
import matplotlib.pyplot as plt

eig_val = np.load("eigen_val.npy")
print(eig_val)
k = np.linspace(-1e-7,1e-7,num=1000)
#print(k)

plt.hist(eig_val, bins=k)
plt.show()