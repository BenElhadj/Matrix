import sys
sys.path.insert(0, '../ex00')
from vector_matrix import Vector, Matrix

def mul_vec(a: Matrix, u: Vector) -> Vector:
    """Multiply a matrix by a vector and return the resulting vector."""
    if a.shape()[1] != u.shape():
        print("Cannot multiply matrix by vector of different shapes.")
        return
    result = [0.0] * a.shape()[0]
    for i in range(a.shape()[0]):
        for j in range(a.shape()[1]):
            result[i] += a.data[i][j] * u.data[j]
    return Vector(result)

def mul_mat(a: Matrix, b: Matrix) -> Matrix:
    """Multiply two matrices and return the resulting matrix."""
    if a.shape()[1] != b.shape()[0]:
        print("Cannot multiply matrices of incompatible shapes.")
        return
    result = Matrix(col=b.shape()[1], row=a.shape()[0])
    for i in range(a.shape()[0]):
        for j in range(b.shape()[1]):
            for k in range(a.shape()[1]):
                result.data[i][j] += a.data[i][k] * b.data[k][j]
    return result

if __name__ == "__main__":
    import numpy as np

    u = Matrix([[0., 0.], [0., 0.]])
    v = Vector([4., 2.])
    print(f"\n[[0, 0], [0, 0]] and [4, 2] --- my mul_vec give ---> {mul_vec(u, v).data}")
    print(f"[[0, 0], [0, 0]] and [4, 2] --- numpy dot give ----> {np.array(u.data) @ np.array(v.data)}")
    # Output: [0. 0.]

    u = Matrix([[1., 0.], [0., 1.]])
    v = Vector([4., 2.])
    print(f"\n[[1, 0], [0, 1]] and [4, 2] --- my mul_vec give ---> {mul_vec(u, v).data}")
    print(f"[[1, 0], [0, 1]] and [4, 2] --- numpy dot give ----> {np.array(u.data) @ np.array(v.data)}")
    # Output: [4. 2.]

    u = Matrix([[1., 1.], [1., 1.]])
    v = Vector([4., 2.])
    print(f"\n[[1, 1], [1, 1]] and [4, 2] --- my mul_vec give ---> {mul_vec(u, v).data}")
    print(f"[[1, 1], [1, 1]] and [4, 2] --- numpy dot give ----> {np.array(u.data) @ np.array(v.data)}")
    # Output: [6, 6]

    u = Matrix([[2., 0.], [0., 2.]])
    v = Vector([2., 1.])
    print(f"\n[[2, 0], [0, 2]] and [2, 1] --- my mul_vec give ---> {mul_vec(u, v).data}")
    print(f"[[2, 0], [0, 2]] and [2, 1] --- numpy dot give ----> {np.array(u.data) @ np.array(v.data)}")
    # Output: [4, 2]
    
    u = Matrix([[0.5, 0.], [0., 0.5]])
    v = Vector([4., 2.])
    print(f"\n[[0.5, 0], [0, 0.5]] and [4, 2] --- my mul_vec give ---> {mul_vec(u, v).data}")
    print(f"[[0.5, 0], [0, 0.5]] and [4, 2] --- numpy dot give ----> {np.array(u.data) @ np.array(v.data)}")
    # Output: [2, 1]