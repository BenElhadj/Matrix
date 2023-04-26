import sys
sys.path.insert(0, '../ex00')
from vector_matrix import Vector

def dot_product(v1: Vector, v2: Vector) -> float:
    """Calculate the dot product between two vectors."""
    if v1.shape() != v2.shape():
        print("Cannot take dot product of vectors with different shapes.")
        return
    return sum(x * y for x, y in zip(v1.data, v2.data))

if __name__ == "__main__":
    import numpy as np
    
    
    u = Vector([0., 0.])
    v = Vector([0., 0.])
    print(f"--- '[0, 0]' and '[0, 0]' --- my dot_product --- give --- {dot_product(u, v)}")
    print(f"--- '[0, 0]' and '[0, 0]' --- numpy.dot -------- give --- {np.dot(u.data, v.data)}\n")
    # 0.0

    u = Vector([1., 0.])
    v = Vector([0., 0.])
    print(f"--- '[1, 0]' and '[0, 0]' --- my dot_product --- give --- {dot_product(u, v)}")
    print(f"--- '[1, 0]' and '[0, 0]' --- numpy.dot -------- give --- {np.dot(u.data, v.data)}\n")
    # 0.0

    u = Vector([1., 0.])
    v = Vector([1., 0.])
    print(f"--- '[1, 0]' and '[1, 0]' --- my dot_product --- give --- {dot_product(u, v)}")
    print(f"--- '[1, 0]' and '[1, 0]' --- numpy.dot -------- give --- {np.dot(u.data, v.data)}\n")
    # 1.0

    u = Vector([1., 0.])
    v = Vector([0., 1.])
    print(f"--- '[1, 0]' and '[0, 1]' --- my dot_product --- give --- {dot_product(u, v)}")
    print(f"--- '[1, 0]' and '[0, 1]' --- numpy.dot -------- give --- {np.dot(u.data, v.data)}\n")   
    # 0.0

    u = Vector([1., 1.])
    v = Vector([1., 1.])
    print(f"--- '[1, 1]' and '[1, 1]' --- my dot_product --- give --- {dot_product(u, v)}")
    print(f"--- '[1, 1]' and '[1, 1]' --- numpy.dot -------- give --- {np.dot(u.data, v.data)}\n")   
    # 2.0
    
    u = Vector([4., 2.])
    v = Vector([2., 1.])
    print(f"--- '[4, 2]' and '[2, 1]' --- my dot_product --- give --- {dot_product(u, v)}")
    print(f"--- '[4, 2]' and '[2, 1]' --- numpy.dot -------- give --- {np.dot(u.data, v.data)}\n")   
    # 10.0

    # # test with vectors of different dimensions
    # u = Vector([1., 2.])
    # v = Vector([1., 2., 3.])
    # print('---------- different dimensions !!! ----------\n')
    # print(f"--- '[1., 2.]' and '[1., 2., 3.]' --- my dot_product --- give --- {dot_product(u, v)}")
    # print(f"--- '[1., 2.]' and '[1., 2., 3.]' --- numpy.dot -------- give --- {np.dot(u.data, v.data)}\n")
    