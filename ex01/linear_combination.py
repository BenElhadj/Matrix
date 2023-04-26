import sys
sys.path.insert(0, '../ex00')
from vector_matrix import Vector

def linear_combination(u, coefs):
    if len(u) != len(coefs):
        print("Input arrays must have the same length.")
        return
    dim = len(u[0])
    return [sum(coef * v[j] for v, coef in zip(u, coefs)) for j in range(dim)]

if __name__ == "__main__":
    import numpy as np

    u1 = [-42., 42.]
    v1 = [-1]
    print(f"\n([-42., 42.]), [-1.] --- my linear_combination --- gives --- {linear_combination([u1], v1)}")
    print(f"([-42., 42.]), [-1.] --------- numpy.dot --------- gives --- {np.dot(np.array(u1).reshape(2, 1), v1)}\n")
    # [42., -42.]

    u2_1 = [-42.]
    u2_2 = [-42.]
    u2_3 = [-42.]
    v2 = [-1., 1., 0.]
    print(f"([-42.][-42.][-42.]), [-1., 1., 0.] --- my linear_combination --- gives --- {linear_combination([u2_1, u2_2, u2_3], v2)}")
    print(f"([-42.][-42.][-42.]), [-1., 1., 0.] --------- numpy.dot --------- gives --- {np.dot(np.transpose((u2_1, u2_2, u2_3)) , np.array(v2).T)}\n")
    # [0.]

    u3_1 = [-42., 42.]
    u3_2 = [1., 3.]
    u3_3 = [10., 20.]
    v3 = [1., -10., -1.]
    print(f"([-42., 42.][1., 3.][10., 20.]), [1., -10., -1.] --- my linear_combination --- gives --- {linear_combination([u3_1, u3_2, u3_3], v3)}")
    print(f"([-42., 42.][1., 3.][10., 20.]), [1., -10., -1.] --------- numpy.dot --------- gives --- {np.dot(np.transpose((u3_1, u3_2, u3_3)) , np.array(v3).T)}\n")
    # [-62., -8.]

    u4_1 = [-42., 100., -69.5]
    u4_2 = [1., 3., 5.]
    v4 = [1., -10.]
    print(f"([-42., 100., -69.5][1., 3., 5.]), [1., -10.] --- my linear_combination --- gives --- {linear_combination([u4_1, u4_2], v4)}")
    print(f"([-42., 100., -69.5][1., 3., 5.]), [1., -10.] --------- numpy.dot --------- gives --- {np.dot(np.transpose((u4_1, u4_2)) , np.array(v4).T)}\n")
    # [-52., 70., -119.5]
