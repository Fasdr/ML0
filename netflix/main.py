import numpy as np
import kmeans
import common
import naive_em
import em


# X = np.loadtxt("toy_data.txt")

# TODO: Your code here
#
# for k in range(1, 5, 1):
#     for i in range(1):
#         gaussian, post = common.init(X, k, seed=i)
#         gaussian, post, new_ll = kmeans.run(X, gaussian, post)
#         common.plot(X, gaussian, post, "K-means: number of classes{}, random seed {}".format(k, i))
#
# for k in range(1, 5, 1):
#     for i in range(1):
#         gaussian, post = common.init(X, k, seed=i)
#         gaussian, post, new_ll = naive_em.run(X, gaussian, post)
#         common.plot(X, gaussian, post, "EM: number of classes{}, random seed {}".format(k, i))

X = np.loadtxt("netflix_incomplete.txt")

# for k in [1, 12]:
#     for i in range(5):
#         gaussian, post = common.init(X, k, seed=i)
#         gaussian, post, new_ll = em.run(X, gaussian, post)
#         print("EM: number of classes {}, random seed {}:".format(k, i))
#         print(new_ll)

# gaussian, post = common.init(X, 12, seed=1)
# gaussian, post, new_ll = em.run(X, gaussian, post)
# print(new_ll)


# for k in range(1, 5, 1):
#     for i in range(5):
#         gaussian, post = common.init(X, k, seed=i)
#         gaussian, post, new_ll = naive_em.run(X, gaussian, post)
#         print("BIC = {} for K = {} and seed = {}".format(common.bic(X, gaussian, new_ll), k, i))
#
#

