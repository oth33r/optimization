import numpy as np
from numpy.linalg import norm


def simplex_method(C, A, b, accuracy):
    # Convert inputs to numpy arrays
    C = np.array(C)
    A = np.array(A)
    b = np.array(b)

    # Number of variables and constraints
    num_vars = len(C)
    num_constraints = len(b)

    # Add slack variables to convert inequalities to equalities
    A = np.hstack((A, np.eye(num_constraints)))
    C = np.hstack((C*(-1), np.zeros(num_constraints)))

    # Initializing array for storing indexes of swapped basic variables
    answer = np.array([-1] * num_vars)

    # Initial tableau
    temp1 = np.hstack((C, np.array([0])))
    temp2 = np.hstack((A, np.vstack(b)))
    tableau = np.vstack((temp1, temp2))

    while True:
        # Determine pivot column
        pivot_col = np.argmin(tableau[0, :-1])
        if abs(tableau[0, pivot_col]) <= accuracy:
            break

        # Determine rate col
        np.seterr(divide='ignore')
        rate = tableau[1:, -1] / tableau[1:, pivot_col]

        # Determine pivot row
        pivot_row = np.where(rate > 0, rate, np.inf).argmin() + 1
        if abs(tableau[pivot_row, pivot_col]) <= accuracy:
            return "The method is not applicable!"

        # Storing indexes of swapped basic variables
        answer[pivot_row - 1] = pivot_col

        # Update tableau
        pivot = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot
        for i in range(len(tableau)):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    # Extract result and objective function value
    x = tableau[1:, -1]
    max_value = tableau[0, -1]
    return x, max_value, answer


def interior_point(C, A, accuracy, init, alpha):
    x = np.array(init, float)
    A = np.array(A)
    c = np.array(C)

    # Number of variables and constraints
    num_vars = len(c)
    num_constraints = len(b)

    # Add slack variables to convert inequalities to equalities
    A = np.hstack((A, np.eye(num_constraints)))
    c = np.hstack((c, np.zeros(num_constraints)))  # maximization

    for i in range(num_vars):
        if c[i] != 0 and x[i] == 0:
            return "The method is not applicable!"

    i = 1
    while True:
        D = np.identity(4) * x
        A_hat = A @ D
        c_hat = D @ c
        P = np.identity(4) - A_hat.transpose() @ np.linalg.inv(A_hat @ A_hat.transpose()) @ A_hat
        c_p = P @ c_hat
        x_hat = np.array([1, 1, 1, 1], float) + (alpha / abs(min(c_p))) * c_p
        # print("In iteration:", i, " we have x =", x)
        i = i + 1
        temp1 = x_hat @ D
        temp2 = x
        temp3 = norm(np.subtract(x_hat @ D, x), ord=2)
        if norm(np.subtract(x_hat @ D, x), ord=2) < accuracy * 0.1:
            break
        if i > 100:
            return "The problem does not have solution!"
        x = x_hat @ D
    # print("In the last iteration", i, "we have x=\n   ", x)
    max_value = x @ c.transpose()
    return x, max_value


with open('input.txt', 'r') as file:
    lines = file.read().split('\n')
i = 0
while i < len(lines):
    # Reading input values
    A = []
    C = [int(num) for num in lines[i].strip().split()]
    i += 2
    while len(lines[i]) != 0:
        A.append([int(num) for num in lines[i].strip().split()])
        i += 1
    i += 1
    b = [int(num) for num in lines[i].strip().split()]
    i += 2
    init = [int(num) for num in lines[i].strip().split()]
    i += 2
    n = int(lines[i].strip())
    accuracy = 10 ** (-n)

    print("C = ", C)
    print("A = ", A)
    print("b = ", b)
    print("init = ", init)
    print("acc = ", accuracy)
    # Applying simplex method
    result_interior1 = interior_point(C, A, accuracy, init, 0.5)
    result_interior2 = interior_point(C, A, accuracy, init, 0.9)
    result_simplex = simplex_method(C, A, b, accuracy)
    # Print result
    if isinstance(result_interior1, str):
        print(result_interior1)
    else:
        x, max_value = result_interior1
        vector = x[:len(C)]
        print("Interior-Point algorithm when α = 0.5:")
        print("Decision variables:", list(np.around(vector, n)))
        print("Maximum value of objective function:", np.around(max_value, n))
        print()

    if isinstance(result_interior2, str):
        print(result_interior2)
    else:
        x, max_value = result_interior2
        vector = x[:len(C)]
        print("Interior-Point algorithm when α = 0.9:")
        print("Decision variables:", list(np.around(vector, n)))
        print("Maximum value of objective function:", np.around(max_value, n))
        print()

    if isinstance(result_simplex, str):
        print(result_simplex)
    else:
        x, max_value, answer = result_simplex
        vector = np.array([0.0] * len(answer))
        for j in range(len(answer)):
            if answer[j] != -1:
                vector[answer[j]] = x[j]
        print("Simplex method from programming Task 1:")
        print("Decision variables:", list(np.around(vector, n)))
        print("Maximum value of objective function:", np.around(max_value, n))
    print()
    i += 2
