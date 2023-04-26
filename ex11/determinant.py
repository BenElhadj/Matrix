import sys
sys.path.insert(0, '../ex00')
from vector_matrix import Matrix

def determinant(matrix: Matrix) -> float:
    if not matrix.is_square():
        print("The matrix must be square")
        return

    rows, cols = matrix.shape()

    if rows == 1:
        return matrix.data[0][0]

    elif rows == 2:
        return matrix.data[0][0] * matrix.data[1][1] - matrix.data[0][1] * matrix.data[1][0]

    elif rows == 3:
        return (
            matrix.data[0][0] * matrix.data[1][1] * matrix.data[2][2]
            + matrix.data[0][1] * matrix.data[1][2] * matrix.data[2][0]
            + matrix.data[0][2] * matrix.data[1][0] * matrix.data[2][1]
            - matrix.data[0][2] * matrix.data[1][1] * matrix.data[2][0]
            - matrix.data[0][1] * matrix.data[1][0] * matrix.data[2][2]
            - matrix.data[0][0] * matrix.data[1][2] * matrix.data[2][1]
        )

    elif rows == 4:
        result = 0
        for i in range(4):
            minor_matrix_data = [[matrix.data[row][col] for col in range(cols) if col != i]
                for row in range(1, rows)]
            result += matrix.data[0][i] * determinant(Matrix(minor_matrix_data)) * (-1) ** i
        return result
    
    else:
        print("Matrix with dimension greater than 4 are not supported")
        return

if __name__ == "__main__":
    import numpy as np
    
    u = Matrix([[0., 0.], [0., 0.]])
    print(f"\n[[0, 0], [0, 0]] --- my determinant give ---> {determinant(u)}")
    print(f"[[0, 0], [0, 0]] --- numpy det give --------> {np.round(np.linalg.det(u.data))}")
    # Output: 0
    
    u = Matrix([[1., 0.], [0., 1.]])
    print(f"\n[[1, 0], [0, 1]] --- my determinant give ---> {determinant(u)}")
    print(f"[[1, 0], [0, 1]] --- numpy det give --------> {np.round(np.linalg.det(u.data))}")
    # Output: 1
     
    u = Matrix([[2., 0.], [0., 2.]])
    print(f"\n[[2, 0], [0, 2]] --- my determinant give ---> {determinant(u)}")
    print(f"[[2, 0], [0, 2]] --- numpy det give --------> {np.round(np.linalg.det(u.data))}")
    # Output: 4
    
    u = Matrix([[1., 1.], [1., 1.]])
    print(f"\n[[1, 1], [1, 1]] --- my determinant give ---> {determinant(u)}")
    print(f"[[1, 1], [1, 1]] --- numpy det give --------> {np.round(np.linalg.det(u.data))}")
    # Output: 0
    
    u = Matrix([[0., 1.], [1., 0.]])
    print(f"\n[[0, 1], [1, 0]] --- my determinant give ---> {determinant(u)}")
    print(f"[[0, 1], [1, 0]] --- numpy det give --------> {np.round(np.linalg.det(u.data))}")
    # Output: -1
    
    u = Matrix([[1., 2.], [3., 4.]])
    print(f"\n[[1, 2], [3, 4]] --- my determinant give ---> {determinant(u)}")
    print(f"[[1, 2], [3, 4]] --- numpy det give --------> {np.round(np.linalg.det(u.data))}")
    # Output: -2 
        
    u = Matrix([[-7., 5.], [4., 6.]])
    print(f"\n[[-7, 5], [4, 6]] --- my determinant give ---> {determinant(u)}")
    print(f"[[-7, 5], [4, 6]] --- numpy det give --------> {np.round(np.linalg.det(u.data))}")
    # Output: -62
    
    u = Matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    print(f"\n[[1, 0, 0], [0, 1, 0], [0, 0, 1]] --- my determinant give ---> {determinant(u)}")
    print(f"[[1, 0, 0], [0, 1, 0], [0, 0, 1]] --- numpy det give --------> {np.round(np.linalg.det(u.data))}")
    # Output: 1