import sys
sys.path.insert(0, '../ex00')
from vector_matrix import Vector

def norm_1(v: Vector) -> float:
    """Compute the 1-norm of a vector, Taxicab norm or Manhattan norm.
    Taxicab: absolute sum of the deviations between the coordinates of the two points.
    Manhattan: the distance between points using the sum of the absolute differences of the coordinates."""
    return sum(abs(x) for x in v.data)

def norm(v: Vector) -> float:
    """Compute the 2-norm of a vector, Euclidean norm.
    measures the distance between two points using the square root of the sum of the squared differences of the coordinates."""
    return sum(x**2 for x in v.data)**0.5

def norm_inf(v: Vector) -> float:
    """Compute the infinity-norm of a vector. Chebyshev norm or Supremum norm
    measures the distance between two points using the maximum value of the absolute coordinate differences."""
    return max(abs(x) for x in v.data)

if __name__ == "__main__":
    import numpy as np
    
   
   
    u = Vector([0.])
    print(f"data[0] with :\nmy norm Euclidean give --> {norm(u)} --- my norm_1 Manhattan give --> {norm_1(u)} --- my norm_inf Supremum give --> {norm_inf(u)}")
    print(f"np.norm Euclidean give --> {np.linalg.norm(u.data, ord=2)} --- np.norm Manhattan give ----> {np.linalg.norm(u.data, ord=1)} --- np.norm Supremum give ------> {np.linalg.norm(u.data, ord=np.inf)} \n")
    # 0.0
    
    
    u = Vector([1.])
    print(f"data[1] with :\nmy norm Euclidean give --> {norm(u)} --- my norm_1 Manhattan give --> {norm_1(u)} --- my norm_inf Supremum give --> {norm_inf(u)}")
    print(f"np.norm Euclidean give --> {np.linalg.norm(u.data, ord=2)} --- np.norm Manhattan give ----> {np.linalg.norm(u.data, ord=1)} --- np.norm Supremum give ------> {np.linalg.norm(u.data, ord=np.inf)} \n")
    # 1

    u = Vector([0., 0.])
    print(f"data[0, 0] with :\nmy norm Euclidean give --> {norm(u)} --- my norm_1 Manhattan give --> {norm_1(u)} --- my norm_inf Supremum give --> {norm_inf(u)}")
    print(f"np.norm Euclidean give --> {np.linalg.norm(u.data, ord=2)} --- np.norm Manhattan give ----> {np.linalg.norm(u.data, ord=1)} --- np.norm Supremum give ------> {np.linalg.norm(u.data, ord=np.inf)} \n")
    # 0.0

    u = Vector([1., 0.])
    print(f"data[1, 0] with :\nmy norm Euclidean give --> {norm(u)} --- my norm_1 Manhattan give --> {norm_1(u)} --- my norm_inf Supremum give --> {norm_inf(u)}")
    print(f"np.norm Euclidean give --> {np.linalg.norm(u.data, ord=2)} --- np.norm Manhattan give ----> {np.linalg.norm(u.data, ord=1)} --- np.norm Supremum give ------> {np.linalg.norm(u.data, ord=np.inf)} \n")
    # 1.0
 
    u = Vector([2., 1.])
    print(f"data[2, 1] with :\nmy norm Euclidean give --> {norm(u)} --- my norm_1 Manhattan give --> {norm_1(u)} --- my norm_inf Supremum give --> {norm_inf(u)}")
    print(f"np.norm Euclidean give --> {np.linalg.norm(u.data, ord=2)} --- np.norm Manhattan give ----> {np.linalg.norm(u.data, ord=1)} --- np.norm Supremum give ------> {np.linalg.norm(u.data, ord=np.inf)} \n")
    # 2.236067977
    
    u = Vector([4., 2.])
    print(f"data[4, 2] with :\nmy norm Euclidean give --> {norm(u)} --- my norm_1 Manhattan give --> {norm_1(u)} --- my norm_inf Supremum give --> {norm_inf(u)}")
    print(f"np.norm Euclidean give --> {np.linalg.norm(u.data, ord=2)} --- np.norm Manhattan give ----> {np.linalg.norm(u.data, ord=1)} --- np.norm Supremum give ------> {np.linalg.norm(u.data, ord=np.inf)} \n")
    # 4.472135955   
     
    u = Vector([-4., -2.])
    print(f"data[-4, -2] with :\nmy norm Euclidean give --> {norm(u)} --- my norm_1 Manhattan give --> {norm_1(u)} --- my norm_inf Supremum give --> {norm_inf(u)}")
    print(f"np.norm Euclidean give --> {np.linalg.norm(u.data, ord=2)} --- np.norm Manhattan give ----> {np.linalg.norm(u.data, ord=1)} --- np.norm Supremum give ------> {np.linalg.norm(u.data, ord=np.inf)} \n")
    # 4.472135955    
    
    