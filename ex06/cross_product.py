import sys
sys.path.insert(0, '../ex00')
from vector_matrix import Vector

def cross_product(u: Vector, v: Vector) -> Vector:
    """Computes the cross product of two 3-dimensional vectors."""
    if u.shape() != 3 or v.shape() != 3:
        print("Input vectors must be 3-dimensional.")
        return # type: ignore
    data = [
        u.data[1] * v.data[2] - u.data[2] * v.data[1],
        u.data[2] * v.data[0] - u.data[0] * v.data[2],
        u.data[0] * v.data[1] - u.data[1] * v.data[0]
    ]
    return Vector(data)

if __name__ == "__main__":
    import numpy as np

    # Test de la fonction cross_product
    u = Vector([0., 0., 0.])
    v = Vector([0., 0., 0.])
    print(f"Vector '[0 0 0]' and '[0 0 0]' --- my add give ---> {cross_product(u, v).data}")
    print(f"Vector '[0 0 0]' and '[0 0 0]' --- numpy angle ---> {np.cross(u.data, v.data)}\n")
    # [0.0, 0.0, 0.0]
    
    u = Vector([1., 0., 0.])
    v = Vector([0., 0., 0.])
    print(f"Vector '[1 0 0]' and '[0 0 0]' --- my add give ---> {cross_product(u, v).data}")
    print(f"Vector '[1 0 0]' and '[0 0 0]' --- numpy angle ---> {np.cross(u.data, v.data)}\n")
    # [0.0, 0.0, 0.0]
    
    u = Vector([1., 0., 0.])
    v = Vector([0., 1., 0.])
    print(f"Vector '[1 0 0]' and '[0 1 0]' --- my add give ---> {cross_product(u, v).data}")
    print(f"Vector '[1 0 0]' and '[0 1 0]' --- numpy angle ---> {np.cross(u.data, v.data)}\n")
    # [0.0, 0.0, 1.0]

    u = Vector([8., 7., -4.])
    v = Vector([3., 2., 1.])
    print(f"Vector '[8 7 -4]' and '[3 2 1]' --- my add give ---> {cross_product(u, v).data}")
    print(f"Vector '[8 7 -4]' and '[3 2 1]' --- numpy angle ---> {np.cross(u.data, v.data)}\n")
    # [15.0, -20.0, -5.0]
    
    u = Vector([1., 1., 1.])
    v = Vector([0., 0., 0.])
    print(f"Vector '[1 1 1]' and '[0 0 0]' --- my add give ---> {cross_product(u, v).data}")
    print(f"Vector '[1 1 1]' and '[0 0 0]' --- numpy angle ---> {np.cross(u.data, v.data)}\n")
    # [0.0, 0.0, 0.0]
    
    u = Vector([1., 1., 1.])
    v = Vector([1., 1., 1.])
    print(f"Vector '[1 1 1]' and '[1 1 1]' --- my add give ---> {cross_product(u, v).data}")
    print(f"Vector '[1 1 1]' and '[1 1 1]' --- numpy angle ---> {np.cross(u.data, v.data)}\n")
    # [0.0, 1.0, 0.0]