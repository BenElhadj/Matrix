import sys
sys.path.insert(0, '../ex00')
from vector_matrix import Matrix

def row_echelon(self) -> 'Matrix':
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

if __name__ == "__main__":
    # pip install sympy
    import sympy 

    u = Matrix([[0, 0], [0, 0]])
    print(f"\n[[0, 0], [0, 0]] --- my row_echelon give ---> Matrix({row_echelon(u).data})")
    print(f"[[0, 0], [0, 0]] --- sympy rref give -------> {sympy.Matrix(u.data).rref()[0]}")
    # Output: [[0, 0], [0, 0]]
    
    u = Matrix([[1, 0], [0, 1]])
    print(f"\n[[1, 0], [0, 1]] --- my row_echelon give ---> Matrix({row_echelon(u).data})")
    print(f"[[1, 0], [0, 1]] --- sympy rref give -------> {sympy.Matrix(u.data).rref()[0]}")
    # Output: [[1, 0], [0, 1]]
    
    u = Matrix([[4, 2], [2, 1]])
    print(f"\n[[4, 2], [2, 1]] --- my row_echelon give ---> Matrix({row_echelon(u).data})")
    print(f"[[4, 2], [2, 1]] --- sympy rref give -------> {sympy.Matrix(u.data).rref()[0]}")
    # Output: [[1, 0.5], [0, 0]]
    
    u = Matrix([[-7, 2], [4, 8]])
    print(f"\n[[-7, 2], [4, 8]] --- my row_echelon give ---> Matrix({row_echelon(u).data})")
    print(f"[[-7, 2], [4, 8]] --- sympy rref give -------> {sympy.Matrix(u.data).rref()[0]}")
    # Output: [[1, 0], [0, 1]]
    
    u = Matrix([[1, 2], [4, 8]])
    print(f"\n[[1, 2], [4, 8]] --- my row_echelon give ---> Matrix({row_echelon(u).data})")
    print(f"[[1, 2], [4, 8]] --- sympy rref give -------> {sympy.Matrix(u.data).rref()[0]}")
    # Output: [[1, 2], [0, 0]]