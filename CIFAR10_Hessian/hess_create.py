import numpy as np

hessian1 = np.load("hessian1.npy")
hessian2 = np.load("hessian2.npy")
hessian3 = np.load("hessian3.npy")
hessian4 = np.load("hessian4.npy")
hessian5 = np.load("hessian5.npy")
hessian6 = np.load("hessian6.npy")
hessian7 = np.load("hessian7.npy")
hessian8 = np.load("hessian8.npy")

print(hessian2.shape)

hessian = np.concatenate((hessian1, hessian2, hessian3, hessian4, hessian5, hessian6, hessian7, hessian8),axis = 0)

print(hessian.shape)
print(hessian[1998:2004,1998:2004])

np.save("hessian_f",hessian)
