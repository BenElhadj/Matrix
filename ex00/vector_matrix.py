from typing import List, Tuple

class Vector():
    def __init__(self, data: List[float]):
        self.data = data

    def shape(self) -> int:
        """Return the shape of the vector."""
        return len(self.data)

    def print_vector(self):
        """Print the vector on the standard output."""
        print(self.data)

    def reshape_to_vector(self, new_shape: int) -> 'Vector':
        """Reshapes the given vector into the new shape."""
        rows = len(self.data)
        new_rows, new_cols = new_shape
        if rows != new_rows * new_cols:
            print("Cannot reshape vector with these dimensions.")
            return
        reshaped_vector = []
        start = 0
        for _ in range(new_rows):
            row = self.data[start:start+new_cols]
            reshaped_vector.append(row)
            start += new_cols
        return Vector(reshaped_vector)
    
    def add(self, v: 'Vector') -> None:
        """Add a vector to this vector."""
        if self.shape() != v.shape():
            print("Cannot add vectors of different shapes.")
            return
        self.data = [self.data[i] + v.data[i] for i in range(self.shape())]

    def sub(self, v: 'Vector') -> None:
        """Subtract a vector from this vector."""
        if self.shape() != v.shape():
            print("Cannot subtract vectors of different shapes.")
            return
        self.data = [self.data[i] - v.data[i] for i in range(self.shape())]

    def scl(self, a: float) -> None:
        """Scale this vector by a scalar."""
        self.data = [a * x for x in self.data]

class Matrix():
    def __init__(self, data: List[List[float]] = None, col: int = None, row: int = None):
        if data is not None:
            self.data = data
        elif col is not None and row is not None:
            self.data = [[0.0] * col for _ in range(row)]
        else:
            print("Either data or col and row must be provided.")
            return

    def shape(self) -> Tuple[int, int]:
        """Return the shape of a matrix as a tuple of the form (number of rows, number of columns)"""
        num_rows = len(self.data)
        num_cols = len(self.data[0]) if num_rows > 0 else 0
        return (num_rows, num_cols)

    def is_square(self) -> bool:
        """Return True if the matrix is square (i.e. has the same number of rows and columns), False otherwise."""
        num_rows, num_cols = self.shape()
        return num_rows == num_cols

    def print_matrix(self):
        """Print the matrix on the standard output."""
        print(self.data) 

    def reshape_to_matrix(self, new_shape: int) -> 'Matrix':
        """Reshapes the given matrix into the new shape."""
        rows, cols = self.shape()
        new_rows, new_cols = new_shape
        if rows * cols != new_rows * new_cols:
            raise ValueError("Cannot reshape matrix with these dimensions.")
        flat_matrix = [elem for row in self.data for elem in row]
        reshaped_matrix = []
        start = 0
        for _ in range(new_rows):
            row = flat_matrix[start:start+new_cols]
            reshaped_matrix.append(row)
            start += new_cols
        return Matrix(reshaped_matrix)

    def add(self, m: 'Matrix') -> None:
        """Add a matrix to this matrix."""
        if self.shape() != m.shape():
            print("Cannot add matrices of different shapes.")
            return
        self.data = [[self.data[i][j] + m.data[i][j] for j in range(self.shape()[1])] for i in range(self.shape()[0])]

    def sub(self, m: 'Matrix') -> None:
        """Subtract a matrix from this matrix."""
        if self.shape() != m.shape():
            print("Cannot subtract matrices of different shapes.")
            return
        self.data = [[self.data[i][j] - m.data[i][j] for j in range(self.shape()[1])] for i in range(self.shape()[0])]

    def scl(self, a: float) -> None:
        """Scale this matrix by a scalar."""
        self.data = [[a * x for x in row] for row in self.data]

if __name__ == '__main__':
    import numpy as np
    
    print("Vector------------------------ Add ------------------------------\n")
    
    u = Vector([0, 0])
    v = Vector([0, 0])
    U = u.data
    u.add(v)
    print(f"Vector --- '[0, 0]' and '[0, 0]' --- my add ------ give --- {u.data}")
    print(f"Vector --- '[0, 0]' and '[0, 0]' --- numpy add --- give --- {np.add(U, v.data)}\n")

    u = Vector([1, 0])
    v = Vector([0, 1])
    U = u.data
    u.add(v)
    print(f"Vector --- '[1, 0]' and '[0, 1]' --- my add ------ give --- {u.data}")
    print(f"Vector --- '[1, 0]' and '[0, 1]' --- numpy add --- give --- {np.add(U, v.data)}\n")

    u = Vector([1, 1])
    v = Vector([1, 1])
    U = u.data
    u.add(v)  
    print(f"Vector --- '[1, 1]' and '[1, 1]' --- my add ------ give --- {u.data}")
    print(f"Vector --- '[1, 1]' and '[1, 1]' --- numpy add --- give --- {np.add(U, v.data)}\n")
    
    u = Vector([21, 21])
    v = Vector([21, 21])  
    U = u.data
    u.add(v)  
    print(f"Vector --- '[21, 21]' and '[21, 21]' --- my add ------ give --- {u.data}")
    print(f"Vector --- '[21, 21]' and '[21, 21]' --- numpy add --- give --- {np.add(U, v.data)}\n")

    u = Vector([-21, 21])
    v = Vector([21, -21]) 
    U = u.data
    u.add(v)  
    print(f"Vector --- '[-21, 21]' and '[21, -21]' --- my add ------ give --- {u.data}")
    print(f"Vector --- '[-21, 21]' and '[21, -21]' --- numpy add --- give --- {np.add(U, v.data)}\n")
    
    u = Vector([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    v = Vector([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]) 
    U = u.data
    u.add(v)
    print(f"Vector --- '[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]' and '[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]' --- my add --- give ------ {u.data}")
    print(f"Vector --- '[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]' and '[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]' --- numpy add --- give --- {np.add(U, v.data)}\n")
        
    print("Matrix------------------------ Add ------------------------------\n")
    
    u = Matrix([[0., 0.], [0., 0.]])
    v = Matrix([[0., 0.], [0., 0.]])
    U = u.data
    u.add(v)
    print(f"Matrix --- [[0, 0], [0, 0]]' and '[[0, 0], [0, 0]]' --- my add ------ give --- {u.data}")
    print(f"Matrix --- [[0, 0], [0, 0]]' and '[[0, 0], [0, 0]]' --- numpy add --- give --- {str(np.add(U, v.data).tolist())}\n")

    u = Matrix([[1., 0.], [0., 1.]])
    v = Matrix([[0., 0.], [0., 0.]])
    U = u.data
    u.add(v)
    print(f"Matrix --- '[[1, 0], [0, 1]]' and '[[0, 0], [0, 0]]' --- my add ------ give --- {u.data}")
    print(f"Matrix --- '[[1, 0], [0, 1]]' and '[[0, 0], [0, 0]]' --- numpy add --- give --- {str(np.add(U, v.data).tolist())}\n")
         
    u = Matrix([[1., 1.], [1., 1.]])
    v = Matrix([[1., 1.], [1., 1.]])
    U = u.data
    u.add(v)
    print(f"Matrix --- '[[1, 1], [1, 1]]' and '[[1, 1], [1, 1]]' --- my add ------ give --- {u.data}")
    print(f"Matrix --- '[[1, 1], [1, 1]]' and '[[1, 1], [1, 1]]' --- numpy add --- give --- {str(np.add(U, v.data).tolist())}\n")
    
    u = Matrix([[21, 21], [21, 21]])
    v = Matrix([[21, 21], [21, 21]])
    U = u.data
    u.add(v)
    print(f"Matrix --- '[[21, 21], [21, 21]]' and '[[21, 21], [21, 21]]' --- my add ------ give --- {u.data}")
    print(f"Matrix --- '[[21, 21], [21, 21]]' and '[[21, 21], [21, 21]]' --- numpy add --- give --- {str(np.add(U, v.data).tolist())}\n")

    print("\nVector------------------------ Sub ------------------------------\n")

    u = Vector([0, 0])
    v = Vector([0, 0])
    U = u.data
    u.sub(v)
    print(f"Vector --- '[0, 0]' and '[0, 0]' --- my sub ------ give --- {u.data}")
    print(f"Vector --- '[0, 0]' and '[0, 0]' --- numpy sub --- give --- {np.subtract(U, v.data)}\n")

    u = Vector([1, 0])
    v = Vector([0, 1])
    U = u.data
    u.sub(v)
    print(f"Vector --- '[1, 0]' and '[0, 1]' --- my sub ------ give --- {u.data}")
    print(f"Vector --- '[1, 0]' and '[0, 1]' --- numpy sub --- give --- {np.subtract(U, v.data)}\n")

    u = Vector([1, 1])
    v = Vector([1, 1])
    U = u.data
    u.sub(v)
    print(f"Vector --- '[1, 1]' and '[1, 1]' --- my sub ------ give --- {u.data}")
    print(f"Vector --- '[1, 1]' and '[1, 1]' --- numpy sub --- give --- {np.subtract(U, v.data)}\n")

    u = Vector([21, 21])
    v = Vector([21, 21])
    U = u.data
    u.sub(v)
    print(f"Vector --- '[21, 21]' and '[21, 21]' --- my sub ------ give --- {u.data}")
    print(f"Vector --- '[21, 21]' and '[21, 21]' --- numpy sub --- give --- {np.subtract(U, v.data)}\n")

    u = Vector([-21, 21])
    v = Vector([21, -21])
    U = u.data
    u.sub(v)
    print(f"Vector --- '[-21, 21]' and '[21, -21]' --- my sub ------ give --- {u.data}")
    print(f"Vector --- '[-21, 21]' and '[21, -21]' --- numpy sub --- give --- {np.subtract(U, v.data)}\n")
    
    u = Vector([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    v = Vector([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    U = u.data
    u.sub(v)
    print(f"Vector --- '[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]' and '[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]' --- my sub ------ give --- {u.data}")
    print(f"Vector --- '[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]' and '[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]' --- numpy sub --- give --- {np.subtract(U, v.data)}\n")

    print("Matrix------------------------ Sub ------------------------------\n")
    
    u = Matrix([[0., 0.], [0., 0.]])
    v = Matrix([[0., 0.], [0., 0.]])
    U = u.data
    u.sub(v)
    print(f"Matrix --- [[0, 0], [0, 0]]' and '[[0, 0], [0, 0]]' --- my sub ------ give --- {u.data}")
    print(f"Matrix --- [[0, 0], [0, 0]]' and '[[0, 0], [0, 0]]' --- numpy sub --- give --- {str(np.subtract(U, v.data).tolist())}\n")

    u = Matrix([[1., 0.], [0., 1.]])
    v = Matrix([[0., 0.], [0., 0.]])
    U = u.data
    u.sub(v)
    print(f"Matrix --- '[[1, 0], [0, 1]]' and '[[0, 0], [0, 0]]' --- my sub ------ give --- {u.data}")
    print(f"Matrix --- '[[1, 0], [0, 1]]' and '[[0, 0], [0, 0]]' --- numpy sub --- give --- {str(np.subtract(U, v.data).tolist())}\n")
         
    u = Matrix([[1., 1.], [1., 1.]])
    v = Matrix([[1., 1.], [1., 1.]])
    U = u.data
    u.sub(v)
    print(f"Matrix --- '[[1, 1], [1, 1]]' and '[[1, 1], [1, 1]]' --- my sub ------ give --- {u.data}")
    print(f"Matrix --- '[[1, 1], [1, 1]]' and '[[1, 1], [1, 1]]' --- numpy sub --- give --- {str(np.subtract(U, v.data).tolist())}\n")
    
    u = Matrix([[21, 21], [21, 21]])
    v = Matrix([[21, 21], [21, 21]])
    U = u.data
    u.sub(v)
    print(f"Matrix --- '[[21, 21], [21, 21]]' and '[[21, 21], [21, 21]]' --- my sub ------ give --- {u.data}")
    print(f"Matrix --- '[[21, 21], [21, 21]]' and '[[21, 21], [21, 21]]' --- numpy sub --- give --- {str(np.subtract(U, v.data).tolist())}\n")

    print("\nVector------------------------ scl/multiply ------------------------------\n")

    u = Vector([0, 0])
    v = 1
    U = u.data
    u.scl(v)
    print(f"Vector --- '[0, 0]' and '1' --- my scl ----------- give --- {u.data}")
    print(f"Vector --- '[0, 0]' and '1' --- numpy multiply --- give --- {np.multiply(U, v)}\n")

    u = Vector([1, 0])
    v = 1
    U = u.data
    u.scl(v)
    print(f"Vector --- '[1, 0]' and '1' --- my scl ----------- give --- {u.data}")
    print(f"Vector --- '[1, 0]' and '1' --- numpy multiply --- give --- {np.multiply(U, v)}\n")

    u = Vector([1, 1])
    v = 2
    U = u.data
    u.scl(v)
    print(f"Vector --- '[1, 1]' and '2' --- my scl ----------- give --- {u.data}")
    print(f"Vector --- '[1, 1]' and '2' --- numpy multiply --- give --- {np.multiply(U, v)}\n")
    
    u = Vector([21, 21])
    v = 2  
    U = u.data
    u.scl(v)
    print(f"Vector --- '[21, 21]' and '2' --- my scl ----------- give --- {u.data}")
    print(f"Vector --- '[21, 21]' and '2' --- numpy multiply --- give --- {np.multiply(U, v)}\n")

    u = Vector([42, 42])
    v = 0.5 
    U = u.data
    u.scl(v)
    print(f"Vector --- '[42, 42]' and '0.5' --- my scl ----------- give --- {u.data}")
    print(f"Vector --- '[42, 42]' and '0.5' --- numpy multiply --- give --- {np.multiply(U, v)}\n")

    print("Matrix------------------------ scl/multiply ------------------------------\n")

    u = Matrix([[0., 0.], [0., 0.]])
    v = 0
    U = u.data
    u.scl(v)
    print(f"Vector --- [[0, 0], [0, 0]]' and '0' --- my scl ----------- give --- {u.data}")
    print(f"Vector --- [[0, 0], [0, 0]]' and '0' --- numpy multiply --- give --- {str(np.multiply(U, v).tolist())}\n")

    u = Matrix([[1., 0.], [0., 1.]])
    v = 1
    U = u.data
    u.scl(v)
    print(f"Vector --- '[[1, 0], [0, 1]]' and '1' --- my scl ----------- give --- {u.data}")
    print(f"Vector --- '[[1, 0], [0, 1]]' and '1' --- numpy multiply --- give --- {str(np.multiply(U, v).tolist())}\n")

    u = Matrix([[1., 2.], [3., 4.]])
    v = 2
    U = u.data
    u.scl(v)
    print(f"Vector --- '[[1, 2], [3, 4]]' and '2' --- my scl ----------- give --- {u.data}")
    print(f"Vector --- '[[1, 2], [3, 4]]' and '2' --- numpy multiply --- give --- {str(np.multiply(U, v).tolist())}\n")

    u = Matrix([[21, 21], [21, 21]])
    v = 0.5  
    U = u.data
    u.scl(v)
    print(f"Vector --- '[[21, 21], [21, 21]]' and '0.5' --- my scl ----------- give --- {u.data}")
    print(f"Vector --- '[[21, 21], [21, 21]]' and '0.5' --- numpy multiply --- give --- {str(np.multiply(U, v).tolist())}\n")


    # u = Vector([-21, 21, 78])
    # U = u.data
    
    # print("\nTest the reshape_to_vector method v1 befor ==", end=" ")
    # u.print_vector()
    # print(f"Result the numpy reshape method v1 befor   == {str(np.array(U).tolist())}")
    # v3 = u.reshape_to_vector((3,1))
    # print("Test the reshape_to_vector method v3 After ==", end=" ")
    # v3.print_vector()
    # print(f"Result the numpy reshape method v3 After   == {str(np.reshape(U, [3,1]).tolist())}")
    
    
    # m = Matrix([[1., 2.], [3., 4.],[5., 6.]])
    # M = m.data
    
    
    # print("\nTest the reshape_to_matrix method m1 befor ==", end=" ")
    # m.print_matrix()
    # print(f"Result the numpy reshape method M1 befor   == {str(np.array(M).tolist())}")
    # m3 = m.reshape_to_matrix((2, 3))
    # print("Test the reshape_to_matrix method m3 After ==", end=" ")
    # m3.print_matrix() 
    # print(f"Result the numpy reshape method M3 After   == {str(np.reshape(M, [2,3]).tolist())}")