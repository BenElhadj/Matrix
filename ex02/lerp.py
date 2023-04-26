# import sys
# sys.path.insert(0, '../ex00')
# from vector_matrix import Matrix, Vector

# def lerp_test(u, v, t):
#     return u + (v - u) * t

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
        return [[u[i] * (1 - t) + v[i] * t] for i in range(len(u))]
    else:
        print("Input objects must be numbers or lists.")
        return

if __name__ == "__main__":
    import numpy as np


    print(f"\nVector --- my Linear_interpolation --- (0., 1., 0.) --- gives --- {lerp(0., 1., 0.)}")
    print(f"Vector --- numpy interp -------------- (0., 1., 0.) --- gives --- {np.interp(0., [0., 1.], [0., 1.])}")
    # 0.0
    
    print(f"\nVector --- my Linear_interpolation --- (0., 1., 1.) --- gives --- {lerp(0., 1., 1.)}")
    print(f"Vector --- numpy interp -------------- (0., 1., 1.) --- gives --- {np.interp(1., [0., 1.], [0., 1.])}")
    # 1.0

    print(f"\nVector --- my Linear_interpolation --- (0., 42., 0.5) --- gives --- {lerp(0., 42., 0.5)}")
    print(f"Vector --- numpy interp -------------- (0., 42., 0.5) --- gives --- {np.interp(0.5, [0., 1.], [0., 42.])}")
    # 21.0
   
    print(f"\nVector --- my Linear_interpolation --- (-42., 42., 0.5) --- gives --- {lerp(-42., 42., 0.5)}")
    print(f"Vector --- numpy interp -------------- (-42., 42., 0.5) --- gives --- {np.interp(0.5, [0., 1.], [-42, 42.])}")
    # 0.0

    print(f"\nMatrix --- my Linear_interpolation --- ([-42., 42.][42., -42.]) --- gives --- {lerp([-42., 42.], [42., -42.], 0.5)}")
    print(f"Matrix --- numpy interp -------------- ([-42., 42.][42., -42.]) --- gives --- {str(np.array([[np.interp(0.5, [0., 1.], [-42., 42.])], [np.interp(0.5, [0., 1.], [42., -42.])]]).tolist())}")
    # [0.0] [0.0]
    