import numpy as np

def projection(fov, ratio, near, far):
    # Convertir l'angle fov en radians
    fov_rad = np.deg2rad(fov)

    # Calculer la taille du plan de projection
    top = near * np.tan(fov_rad / 2)
    right = top * ratio

    # Créer la matrice de projection
    proj_matrix = np.zeros((4, 4), dtype=np.float32)
    proj_matrix[0, 0] = near / right
    proj_matrix[1, 1] = near / top
    proj_matrix[2, 2] = -(far + near) / (far - near)
    proj_matrix[2, 3] = -(2 * far * near) / (far - near)
    proj_matrix[3, 2] = -1

    return proj_matrix

def save_matrix_to_file(matrix, filename):
    with open(filename, 'w') as f:
        for row in matrix:
            f.write(", ".join([str(val) for val in row]) + "\n")

if __name__ == "__main__":
    fov = 40
    ratio = 16 / 9
    near = 1
    far = 500

    proj_matrix = projection(fov, ratio, near, far)

    # Enregistrez la matrice de projection dans un fichier nommé "proj.txt"
    save_matrix_to_file(proj_matrix, "proj")

    # Afficher la matrice de projection
    # print("Matrice de projection :")
    # print(proj_matrix)
