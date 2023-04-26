import sys
sys.path.insert(0, '../ex00')
from vector_matrix import Matrix

def transpose(m) -> 'Matrix':
    num_rows, num_cols = m.shape()
    return [[m.data[j][i] for j in range(num_rows)] for i in range(num_cols)]

if __name__ == "__main__":
    import numpy as np
    
    u = Matrix([[0, 0], [0, 0]])
    print(f"\n[[0, 0], [0, 0]] --- my transpose give ---> {transpose(u)}")
    print(f"[[0, 0], [0, 0]] --- np.transpose give ---> {np.array(u.data).T.tolist()}")
    # Output: [[0, 0], [0, 0]]

    u = Matrix([[1, 0], [0, 1]])
    print(f"\n[[1, 0], [0, 1]] --- my transpose give ---> {transpose(u)}")
    print(f"[[1, 0], [0, 1]] --- np.transpose give ---> {np.array(u.data).T.tolist()}")
    # Output: [[1, 0], [0, 1]]

    u = Matrix([[1, 2], [3, 4]])
    print(f"\n[[1, 2], [3, 4]] --- my transpose give ---> {transpose(u)}")
    print(f"[[1, 2], [3, 4]] --- np.transpose give ---> {np.array(u.data).T.tolist()}")
    # Output: [[1, 3], [2, 4]]
    
    u = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    print(f"\n[[1, 0, 0], [0, 1, 0], [0, 0, 1]] --- my transpose give ---> {transpose(u)}")
    print(f"[[1, 0, 0], [0, 1, 0], [0, 0, 1]] --- np.transpose give ---> {np.array(u.data).T.tolist()}")
    # Output: [1, 0, 0], [0, 1, 0], [0, 0, 1]
    
    u = Matrix([[1, 2], [3, 4], [5, 6]])
    print(f"\n[[1, 2], [3, 4], [5, 6]] --- my transpose give ---> {transpose(u)}")
    print(f"[[1, 2], [3, 4], [5, 6]] --- np.transpose give ---> {np.array(u.data).T.tolist()}")
    # Output: [1, 3, 5], [2, 4, 6]