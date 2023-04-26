import math
from typing import List, Tuple
from typing import List, Tuple
import cmath

class ComplexVector():
    def __init__(self, data: List[complex]):
        self.data = data

    def shape(self) -> int:
        """Return the shape of the vector."""
        return len(self.data)

    def print_vector(self):
        """Print the vector on the standard output."""
        print(self.data)

    def reshape_to_vector(self, new_shape: int) -> 'ComplexVector':
        """Reshapes the given vector into the new shape."""
        rows = len(self.data)
        new_rows, new_cols = new_shape
        if rows != new_rows * new_cols:
            print("Cannot reshape vector with these dimensions.")
            return None
        reshaped_vector = []
        start = 0
        for _ in range(new_rows):
            row = self.data[start:start+new_cols]
            reshaped_vector.append(row)
            start += new_cols
        return ComplexVector(reshaped_vector)
    
    def add(self, v: 'ComplexVector') -> None:
        """Add a vector to this vector."""
        if self.shape() != v.shape():
            print("Cannot add vectors of different shapes.")
            return
        self.data = [self.data[i] + v.data[i] for i in range(self.shape())]

    def sub(self, v: 'ComplexVector') -> None:
        """Subtract a vector from this vector."""
        if self.shape() != v.shape():
            print("Cannot subtract vectors of different shapes.")
            return
        self.data = [self.data[i] - v.data[i] for i in range(self.shape())]

    def scl(self, a: float) -> None:
        """Scale this vector by a scalar."""
        self.data = [a * x for x in self.data]

class ComplexMatrix():
    def __init__(self, data: List[List[complex]] = None, col: int = None, row: int = None): # type: ignore
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
        for row in self.data:
            print(row)

    def reshape_to_matrix(self, new_shape: int) -> 'ComplexMatrix':
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
        return ComplexMatrix(reshaped_matrix)

    def add(self, m: 'ComplexMatrix') -> None:
        """Add a matrix to this matrix."""
        if self.shape() != m.shape():
            print("Cannot add matrices of different shapes.")
            return
        self.data = [[self.data[i][j] + m.data[i][j] for j in range(self.shape()[1])] for i in range(self.shape()[0])]

    def sub(self, m: 'ComplexMatrix') -> None:
        """Subtract a matrix from this matrix."""
        if self.shape() != m.shape():
            print("Cannot subtract matrices of different shapes.")
            return
        self.data = [[self.data[i][j] - m.data[i][j] for j in range(self.shape()[1])] for i in range(self.shape()[0])]

    def scl(self, a: float) -> None:
        """Scale this matrix by a scalar."""
        self.data = [[a * x for x in row] for row in self.data]

def linear_combination(u, coefs):
    if len(u) != len(coefs):
        print("Input arrays must have the same length.")
        return
    dim = len(u[0])
    return [sum(coef * v[j] for v, coef in zip(u, coefs)) for j in range(dim)]

def lerp(u, v, t):
    if type(u) != type(v):
        print("Input objects must have the same type.")
        return
    if not (0 <= t <= 1):
        print("t must be between 0 and 1.")
        return
    
    if isinstance(u, (int, float, complex)):
        return u * (1 - t) + v * t
    elif isinstance(u, list):
        if len(u) != len(v):
            print("Input lists must have the same length.")
            return
        return [u[i] * (1 - t) + v[i] * t for i in range(len(u))]
    elif isinstance(u, ComplexMatrix):
        if u.shape() != v.shape():
            print("Input matrices must have the same shape.")
            return
        rows, cols = u.shape()
        return ComplexMatrix([[u.data[i][j] * (1 - t) + v.data[i][j] * t for j in range(cols)] for i in range(rows)])
    else:
        print("Input objects must be numbers or lists.")
        return
    
def c_dot_product(v1: ComplexVector, v2: ComplexVector) -> complex:
    """Calculate the dot product between two complex vectors."""
    if v1.shape() != v2.shape():
        print("Cannot take dot product of vectors with different shapes.")
        return
    return sum(x * y.conjugate() for x, y in zip(v1.data, v2.data))

def norm_1(v: ComplexVector) -> float:
    """Compute the 1-norm of a vector."""
    return sum(abs(x) for x in v.data)

def c_norm(v: ComplexVector) -> float:
    """Compute the 2-norm of a complex vector."""
    return sum(abs(x)**2 for x in v.data)**0.5

def norm_inf(v: ComplexVector) -> float:
    """Compute the infinity-norm of a vector."""
    return max(abs(x) for x in v.data)

def angle_cos(u: ComplexVector, v: ComplexVector) -> float:
    if len(u.data) != len(v.data):
        print("Vectors must have the same size.")
        return
    product = c_dot_product(u, v)
    norm_u = c_norm(u)
    norm_v = c_norm(v)
    if norm_u == 0.0 or norm_v == 0.0:
        print("Vectors must not be zero vectors.")
        return
    return product / (norm_u * norm_v)

def cross_product(u: ComplexVector, v: ComplexVector) -> ComplexVector:
    """Computes the cross product of two 3-dimensional vectors."""
    if u.shape() != 3 or v.shape() != 3:
        print("Input vectors must be 3-dimensional.")
        return
    data = [
        math.fma(u.data[1], v.data[2], -u.data[2] * v.data[1]),
        math.fma(u.data[2], v.data[0], -u.data[0] * v.data[2]),
        math.fma(u.data[0], v.data[1], -u.data[1] * v.data[0])
    ]
    return ComplexVector(data)

def mul_vec(a: ComplexMatrix, u: ComplexVector) -> ComplexVector:
    """Multiply a matrix by a vector and return the resulting vector."""
    if a.shape()[1] != u.shape():
        print("Cannot multiply matrix by vector of different shapes.")
        return
    result = [0.0] * a.shape()[0]
    for i in range(a.shape()[0]):
        for j in range(a.shape()[1]):
            result[i] += a.data[i][j] * u.data[j]
    return ComplexVector(result)

def mul_mat(a: ComplexMatrix, b: ComplexMatrix) -> ComplexMatrix:
    """Multiply two matrices and return the resulting matrix."""
    if a.shape()[1] != b.shape()[0]:
        print("Cannot multiply matrices of incompatible shapes.")
        return
    result = ComplexMatrix(col=b.shape()[1], row=a.shape()[0])
    for i in range(a.shape()[0]):
        for j in range(b.shape()[1]):
            for k in range(a.shape()[1]):
                result.data[i][j] += a.data[i][k] * b.data[k][j]
    return result

def trace(m: ComplexMatrix) -> float:
    """Calcule la somme des éléments de la diagonale d'une matrice carrée."""
    if not m.is_square():
        print("La matrice doit être carrée.")
        return
    return sum(m.data[i][i] for i in range(m.shape()[0]))

def transpose(m) -> 'ComplexMatrix':
    num_rows, num_cols = m.shape()
    return [[m.data[j][i] for j in range(num_rows)] for i in range(num_cols)]

def row_echelon(self) -> 'ComplexMatrix':
    rowCount = len(self.data)
    columnCount = len(self.data[0])
    for lead, r in enumerate(range(rowCount)):
        if lead >= columnCount:
            return self
        i = r
        while abs(self.data[i][lead]) < 1e-8:
            i += 1
            if i == rowCount:
                i = r
                lead += 1
                if columnCount == lead:
                    return self
        self.data[i], self.data[r] = self.data[r], self.data[i]
        lv = self.data[r][lead]
        self.data[r] = [mrx / float(lv) for mrx in self.data[r]]
        for i in range(rowCount):
            if i != r:
                lv = self.data[i][lead]
                self.data[i] = [iv - lv*rv for rv, iv in zip(self.data[r], self.data[i])]
    return self

def determinant(matrix: ComplexMatrix) -> float:
    if not matrix.is_square():
        print("La matrice doit être carrée")
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
            result += matrix.data[0][i] * determinant(ComplexMatrix(minor_matrix_data)) * (-1) ** i
        return result
    
    else:
        print("Les matrices de dimension supérieure à 4 ne sont pas prises en charge")
        return
    
def inverse(m: ComplexMatrix) -> ComplexMatrix:
    n = len(m.data)
    inv = ComplexMatrix(col=n, row=n)
    for i in range(n):
        inv.data[i][i] = 1

    for i in range(n):
        if m.data[i][i] == 0:
            for j in range(i + 1, n):
                if m.data[j][i] != 0:
                    m.data[i], m.data[j] = m.data[j], m.data[i]
                    inv.data[i], inv.data[j] = inv.data[j], inv.data[i]
                    break
            else:
                print("Matrix is singular.")
                return

        pivot = m.data[i][i]
        for j in range(n):
            m.data[i][j] /= pivot
            inv.data[i][j] /= pivot

        for j in range(n):
            if j == i:
                continue
            factor = m.data[j][i]
            for k in range(n):
                m.data[j][k] -= factor * m.data[i][k]
                inv.data[j][k] -= factor * inv.data[i][k]

    return inv

def rank(matrix: ComplexMatrix) -> int:
    m = ComplexMatrix(data=[row.copy() for row in matrix.data])
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
            m.data[row][col] = 1.0
            for i in range(n):
                if i != row:
                    factor = m.data[i][col]
                    for j in range(col + 1, len(m.data[row])):
                        m.data[i][j] -= factor * m.data[row][j]
                    m.data[i][col] = 0.0
            col += 1
            rank += 1
    return rank


if __name__ == "__main__":
    from typing import List
    import numpy as np

    print("Create two vectors\nv1 = ComplexVector([1+2j, 3+4j, 5+6j])\nv2 = ComplexVector([7+8j, 9+10j, 11+12j])")
    v1 = ComplexVector([1+2j, 3+4j, 5+6j])
    V1 =      np.array([1+2j, 3+4j, 5+6j])
    v2 = ComplexVector([7+8j, 9+10j, 11+12j])
    V2 =      np.array([7+8j, 9+10j, 11+12j])
    
    print(f"Test the shape method v1 == {v1.shape()}")
    print(f"Test the shape method v2 == {v2.shape()}\n")
    # Output: 3

    print("Test the print_vector method v1 ==", end=" ")
    v1.print_vector()
    # Output: [1+2j, 3+4j, 5+6j]
    print("Test the print_vector method v2 ==", end=" ")
    v2.print_vector()
    # Output: [7+8j, 9+10j, 11+12j]

    print("\nTest the add method   ==", end=" ")
    v1.add(v2)
    v1.print_vector()
    V1 = np.add(V1, V2)
    print(f"Result add with numpy == {str(V1).replace('.', '')}")
    # Output: [8+10j, 12+14j, 16+18j]
    
    print("\nTest the sub method   ==", end=" ")

    v1.sub(v2)
    v1.print_vector()
    V1 = np.subtract(V1 ,V2)
    print(f"Result sub with numpy == {str(V1).replace('.', '')}") 
    # Output: [1+2j, 3+4j, 5+6j]

    print("\nTest the scl method        ==", end=" ")
    v1.scl(2)
    v1.print_vector()
    V1 = np.multiply(V1, 2)
    print(f"Result multiply with numpy == {str(V1).replace('. ', '')}") 
    # Output: [2+4j, 6+8j, 10+12j]
  
    print("\nTest the reshape_to_vector method v1 befor ==", end=" ")
    v1.print_vector()
    print(f"Result the numpy reshape method v1 befor   == {str(np.array(V1).tolist())}")
    v3 = v1.reshape_to_vector((3,1))
    print("Test the reshape_to_vector method v3 After ==", end=" ")
    v3.print_vector()
    print(f"Result the numpy reshape method v3 After   == {str(np.reshape(V1, [3,1]).tolist())}")
    # Output: [(2+4j), (6+8j), (10+12j)] ==> [[(2+4j)], [(6+8j)], [(10+12j)]]
    
    print("\nCreate two matrices\nm1 = ComplexMatrix([[1+2j, 3+4j, 5+6j], [7+8j, 9+10j, 11+12j]]\nm2 = ComplexMatrix([[13+14j, 15+16j, 17+18j], [19+20j, 21+22j, 23+24j]])")
    m1 = ComplexMatrix([[1+2j, 3+4j, 5+6j], [7+8j, 9+10j, 11+12j]])
    M1 =      np.array([[1+2j, 3+4j, 5+6j], [7+8j, 9+10j, 11+12j]])
    m2 = ComplexMatrix([[13+14j, 15+16j, 17+18j], [19+20j, 21+22j, 23+24j]])
    M2 =      np.array([[13+14j, 15+16j, 17+18j], [19+20j, 21+22j, 23+24j]])
    
    print(f"\nTest the shape method m1 == {m1.shape()}")
    print(f"Test the shape method m2 == {m2.shape()}\n")
    # Output: (2, 3)

    print("Test the print_matrix method m1 ==")
    m1.print_matrix() 
    # Output:
    # [1+2j, 3+4j, 5+6j]
    # [7+8j, 9+10j, 11+12j]
    print("Test the print_matrix method m1 ==")
    m2.print_matrix() 
    # Output:
    # [13+14j, 15+16j, 17+18j]
    # [19+20j, 21+22j, 23+24j]
    
    print("\nTest the add method   ==", end=" ")
    m1.add(m2)
    m1.print_matrix()
    M1 = np.add(M1, M2)
    print(f"Result add with numpy == {str(M1).replace('.', '')}")
    # Output:
    # [14+16j, 18+20j, 22+24j]
    # [26+28j, 30+32j, 34+36j]

    print("\nTest the sub method   ==", end=" ")
    m1.sub(m2)
    m1.print_matrix()
    M1 = np.subtract(M1 ,M2)
    print(f"Result sub with numpy == {str(M1).replace('. ', '').replace('.', '')}") 
    # Output:
    # [1+2j, 3+4j, 5+6j]
    # [7+8j, 9+10j, 11+12j]

    print("\nTest the reshape_to_matrix method m1 befor ==", end=" ")
    m1.print_matrix()
    print(f"Result the numpy reshape method M1 befor   == {str(np.array(M1)).replace('. ', '').replace('.', '')}")
    m3 = m1.reshape_to_matrix((3, 2))
    print("Test the reshape_to_matrix method m3 After ==", end=" ")
    m3.print_matrix() 
    print(f"Result the numpy reshape method M3 After   == {str(np.reshape(M1, [3,2])).replace('. ', '').replace('.', '')}")
    # Test the reshape_to_matrix method
    # Output:
    # [2+4j, 6+8j]
    # [10+12j, 14+16j]
    # [18+20j, 22+24j]

    print("\nTest the scl method        ==", end=" ")
    m1.scl(2)
    m1.print_matrix() 
    M1 = np.multiply(M1, 2)
    print(f"Result multiply with numpy == {str(M1).replace('. ', '').replace('.', '')}") 
    # Output:
    # [2+4j, 6+8j, 10+12j]
    # [14+16j, 18+20j, 22+24j]