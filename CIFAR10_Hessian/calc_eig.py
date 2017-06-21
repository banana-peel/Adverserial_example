import numpy as np 



hessian = load("hessian_f")
eig_val,eig_vec = np.linalg.eig(hessian)
np.save("eigen_val",eig_val)
np.save("eigen_vec",eig_vec)
