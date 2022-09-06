import numpy as np

# Exercise 6.1 b
P = np.array([[0.5, 0.5, 0, 0, 0, 0],
    [0.5, 0, 0.5, 0, 0, 0],
    [0, 0.5, 0, 0.5, 0, 0],
    [0, 0, 0.5, 0, 0.5, 0],
    [0, 0, 0, 0.5, 0, 0.5],
    [0, 0, 0, 0, 0.5, 0.5]])

print(f'One step transition matrix: \n {P}')

# Exercise 6.1 d
x0 = np.array([0, 0, 1, 0, 0, 0])
P2 = np.linalg.matrix_power(P, 2)
P4 = np.linalg.matrix_power(P, 4)
P10 = np.linalg.matrix_power(P, 10)
P20 = np.linalg.matrix_power(P, 20)

x2 = np.matmul(x0, P2)
x4 = np.matmul(x0, P4)
x10 = np.matmul(x0, P10)
x20 = np.matmul(x0, P20)

print(f'x2:\n {x2} \n x4:\n {x4} \n x10\n {x10} \n x20:\n {x20}')

p40 = np.linalg.matrix_power(P,40)
print(f'Converged matrix at p40\n {p40}')
