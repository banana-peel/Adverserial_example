import numpy as np
import matplotlib.pyplot as plt

hessian = np.load("eig_val_list.npy")

# for i in range(32):
eig_val = hessian[0]


	# #print(k)
	# print(np.std(eig_val))

k = np.linspace(-1e-9,1e-5,num=1000)
plt.hist(eig_val, bins=k)
plt.title("Histogram of eigen values for Image: 0")
plt.show()