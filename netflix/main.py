import numpy as np
import kmeans
import common
import naive_em
import em


X = np.loadtxt("toy_data.txt")

# TODO: Your code here
#
# for k in range(1, 5, 1):
#     for i in range(5):
#         gaussian, post = common.init(X, k, seed=i)
#         gaussian, post, new_ll = kmeans.run(X, gaussian, post)
#         common.plot(X, gaussian, post, "K-means: number of classes{}, random seed {}".format(k, i))
#
# for k in range(1, 5, 1):
#     for i in range(5):
#         gaussian, post = common.init(X, k, seed=i)
#         gaussian, post, new_ll = naive_em.run(X, gaussian, post)
#         common.plot(X, gaussian, post, "EM: number of classes{}, random seed {}".format(k, i))
#
