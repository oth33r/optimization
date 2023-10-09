import numpy as np


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


# Read the input values from the input.txt file
with open('input.txt', 'r') as file:
    lines = file.read().split('\n')
i = 0
result = ""
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
    n = int(lines[i].strip())
    accuracy = 10**(-n)

    print("C = ", C)
    print("A = ", A)
    print("b = ", b)
    print("acc = ", accuracy)
    # Applying simplex method
    result = simplex_method(C, A, b, accuracy)
    # Print result
    if isinstance(result, str):
        print(result)
    else:
        x, max_value, answer = result
        vector = np.array([0.0] * len(answer))
        for j in range(len(answer)):
            if answer[j] != -1:
                vector[answer[j]] = x[j]
        print("Decision variables:", list(np.around(vector, n)))
        print("Maximum value of objective function:", np.around(max_value, n))
    i += 2