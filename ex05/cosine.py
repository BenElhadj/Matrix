import sys
sys.path.insert(0, '../ex00')
sys.path.insert(0, '../ex03')
sys.path.insert(0, '../ex04')
from vector_matrix import Vector
from dot_product import dot_product
from norm import norm

def angle_cos(u: Vector, v: Vector) -> float:
    if len(u.data) != len(v.data):
        print("Vectors must have the same size.")
        return
    product = dot_product(u, v)
    norm_u = norm(u)
    norm_v = norm(v)
    if norm_u == 0.0 or norm_v == 0.0:
        print("Vectors must not be zero vectors.")
        return
    return product / (norm_u * norm_v)

if __name__ == "__main__":
    import numpy as np


    u = Vector([1.0, 0.0])
    v = Vector([0.0, 1.0])
    print(f"Vector '[1 0]' and '[0 1]' --- my add give ---> {angle_cos(u, v)}")
    print(f"Vector '[1 0]' and '[0 1]' --- numpy angle ---> {np.dot(u.data, v.data) / (np.linalg.norm(u.data) * np.linalg.norm(v.data))}\n")
    # 0.0
    
    u = Vector([8.0, 7.0])
    v = Vector([3.0, 2.0])
    print(f"Vector '[8 7]' and '[3 2]' --- my add give ---> {angle_cos(u, v)}")
    print(f"Vector '[8 7]' and '[3 2]' --- numpy angle ---> {np.dot(u.data, v.data) / (np.linalg.norm(u.data) * np.linalg.norm(v.data))}\n")
    # 0.9914542955425437
     
    u = Vector([1.0, 1.0])
    v = Vector([1.0, 1.0])
    print(f"Vector '[1 1]' and '[1 1]' --- my add give ---> {angle_cos(u, v)}")
    print(f"Vector '[1 1]' and '[1 1]' --- numpy angle ---> {np.dot(u.data, v.data) / (np.linalg.norm(u.data) * np.linalg.norm(v.data))}\n")
    # 1.0
      
    u = Vector([4.0, 2.0])
    v = Vector([1.0, 1.0])
    print(f"Vector '[4 2]' and '[1 1]' --- my add give ---> {angle_cos(u, v)}")
    print(f"Vector '[4 2]' and '[1 1]' --- numpy angle ---> {np.dot(u.data, v.data) / (np.linalg.norm(u.data) * np.linalg.norm(v.data))}\n")
    # 0.9486832980505138
    
    u = Vector([-7.0, 3.0])
    v = Vector([6.0, 4.0])
    print(f"Vector '[-7 3]' and '[6 4]' --- my add give ---> {angle_cos(u, v)}")
    print(f"Vector '[-7 3]' and '[6 4]' --- numpy angle ---> {np.dot(u.data, v.data) / (np.linalg.norm(u.data) * np.linalg.norm(v.data))}\n")
    # -0.5462677805469223
