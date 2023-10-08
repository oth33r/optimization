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

    # Initial tableau
    temp1 = np.hstack((C, np.array([0])))
    temp2 = np.hstack((A, np.vstack(b)))
    answer = np.array([-1]*num_vars)
    tableau = np.vstack((temp1, temp2))

    while True:
        # Determine pivot column
        pivot_col = np.argmin(tableau[0, :-1])
        if abs(tableau[0, pivot_col]) <= accuracy:
            break

        # Determine pivot row
        tempor1 = tableau[1:, -1]
        tempor2 = tableau[1:, pivot_col]

        rate = tableau[1:, -1] / tableau[1:, pivot_col]

        pivot_row = np.where(rate > 0, rate, np.inf).argmin() + 1
        if abs(tableau[pivot_row, pivot_col]) <= accuracy:
            return "The method is not applicable!"
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


# Example inputs
"""
C = [1, 2, 3]
A = [[1, -1, 1], [3, 2, 1], [2, 1, -2]]
b = [2, 5, 1]
n = 6
accuracy = 10**(-n)
print(accuracy)
"""
# Read the input values from the input.txt file
with open('input.txt', 'r') as file:
    lines = file.read().split('\n')
i = 0
result = ""
while i < len(lines):
    A = []
    C = [int(num) for num in lines[i].strip().split()]
    i += 2
    while len(lines[i]) != 0:
        A.append([int(num) for num in lines[i].strip().split()])
        i += 1
    i += 1
    b = [int(num) for num in lines[i].strip().split()]
    i += 2
    accuracy = 10**(-int(lines[i].strip()))

    print("C = ", C)
    print("A = ", A)
    print("b = ", b)
    print("acc = ", accuracy)
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
        print("Decision variables:", vector)
        print("Maximum value of objective function:", max_value)
    i += 2