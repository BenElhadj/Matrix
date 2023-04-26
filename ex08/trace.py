import sys
sys.path.insert(0, '../ex00')
from vector_matrix import Matrix

def trace(m: Matrix) -> float:
    """Calculates the sum of the elements of the diagonal of a square matrix."""
    if not m.is_square():
        print("The matrix must be square.")
        return
    return sum(m.data[i][i] for i in range(m.shape()[0]))

if __name__ == "__main__":
    import numpy as np

    u = Matrix([[0., 0.], [0., 0.]])
    print(f"\n[[0, 0], [0, 0]] --- my trace give ---> {trace(u)}")
    print(f"[[0, 0], [0, 0]] --- np.trace give ---> {np.trace(u.data)}")
    # Output: 0.0

    u = Matrix([[1, 0], [0, 1]])
    print(f"\n[[1, 0], [0, 1]] --- my trace give ---> {trace(u)}")
    print(f"[[1, 0], [0, 1]] --- np.trace give ---> {np.trace(u.data)}")
    # Output: 2.0

    u = Matrix([[1, 2], [3, 4]])
    print(f"\n[[1, 2], [3, 4]] --- my trace give ---> {trace(u)}")
    print(f"[[1, 2], [3, 4]] --- np.trace give ---> {np.trace(u.data)}")
    # Output: 5.0

    u = Matrix([[8, -7], [4, 2]])
    print(f"\n[[8, -7], [4, 2]] --- my trace give ---> {trace(u)}")
    print(f"[[8, -7], [4, 2]] --- np.trace give ---> {np.trace(u.data)}")
    # Output: 10.0

    u = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    print(f"\n[[1, 0, 0], [0, 1, 0], [0, 0, 1]] --- my trace give ---> {trace(u)}")
    print(f"[[1, 0, 0], [0, 1, 0], [0, 0, 1]] --- np.trace give ---> {np.trace(u.data)}")
    # Output: 3.0