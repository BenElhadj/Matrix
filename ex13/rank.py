import sys
sys.path.insert(0, '../ex00')
from vector_matrix import Matrix

def rank(matrix: Matrix) -> int:
    m = Matrix(data=[row.copy() for row in matrix.data])
    n, _ = m.shape()
    rank = 0
    col = 0
    for row in range(n):
        if col >= len(m.data[row]):
            break
        pivot = row
        while m.data[pivot][col] == 0:
            pivot += 1
            if pivot == n:
                pivot = row
                col += 1
                if col == len(m.data[row]):
                    break
        if col < len(m.data[row]):
            m.data[pivot], m.data[row] = m.data[row], m.data[pivot]
            pivot_value = m.data[row][col]
            m.data[row] = [val / pivot_value for val in m.data[row]]
            for i in range(n):
                if i != row:
                    factor = m.data[i][col]
                    for j in range(col + 1, len(m.data[row])):
                        m.data[i][j] -= factor * m.data[row][j]
                    m.data[i][col] = 0.0
            col += 1
            rank += 1
    return rank

if __name__ == '__main__':
    import numpy as np

    u = Matrix([[0, 0], [0, 0]])
    print(f"\n[[0, 0], [0, 0]] --- my rank give ----------> {rank(u)}")
    print(f"[[0, 0], [0, 0]] --- np.matrix_rank give ---> {np.linalg.matrix_rank(u.data)}")
    # Output: 0

    u = Matrix([[1, 0], [0, 1]])
    print(f"\n[[1, 0], [0, 1]] --- my rank give ----------> {rank(u)}")
    print(f"[[1, 0], [0, 1]] --- np.matrix_rank give ---> {np.linalg.matrix_rank(u.data)}")
    # Output: 2
 
    u = Matrix([[2, 0], [0, 2]])
    print(f"\n[[2, 0], [0, 2]] --- my rank give ----------> {rank(u)}")
    print(f"[[2, 0], [0, 2]] --- np.matrix_rank give ---> {np.linalg.matrix_rank(u.data)}")
    # Output: 2

    u = Matrix([[1, 1], [1, 1]])
    print(f"\n[[1, 1], [1, 1]] --- my rank give ----------> {rank(u)}")
    print(f"[[1, 1], [1, 1]] --- np.matrix_rank give ---> {np.linalg.matrix_rank(u.data)}")
    # Output: 1 
   
    u = Matrix([[0, 1], [1, 0]])
    print(f"\n[[0, 1], [1, 0]] --- my rank give ----------> {rank(u)}")
    print(f"[[0, 1], [1, 0]] --- np.matrix_rank give ---> {np.linalg.matrix_rank(u.data)}")
    # Output: 2

    u = Matrix([[1, 2], [3, 4]])
    print(f"\n[[1, 2], [3, 4]] --- my rank give ----------> {rank(u)}")
    print(f"[[1, 2], [3, 4]] --- np.matrix_rank give ---> {np.linalg.matrix_rank(u.data)}")
    # Output: 2
    
    u = Matrix([[-7, 5], [4, 6]])
    print(f"\n[[-7, 5], [4, 6]] --- my rank give ----------> {rank(u)}")
    print(f"[[-7, 5], [4, 6]] --- np.matrix_rank give ---> {np.linalg.matrix_rank(u.data)}")
    # Output: 2

    u = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    print(f"\n[[1, 0, 0], [0, 1, 0], [0, 0, 1]] --- my rank give ----------> {rank(u)}")
    print(f"[[1, 0, 0], [0, 1, 0], [0, 0, 1]] --- np.matrix_rank give ---> {np.linalg.matrix_rank(u.data)}")
    # Output: 3