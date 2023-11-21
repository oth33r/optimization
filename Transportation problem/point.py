import numpy as np

def transportation_problem():
    # Get input from the user
    m = int(input("Enter the number of suppliers: "))
    n = int(input("Enter the number of consumers: "))

    # Input supply vector
    S = np.array([int(x) for x in input("Enter the supply vector (comma-separated): ").split(",")])

    # Input demand vector
    D = np.array([int(x) for x in input("Enter the demand vector (comma-separated): ").split(",")])

    S_copy = np.copy(S)
    D_copy = np.copy(D)
    S_copy2 = np.copy(S)
    D_copy2 = np.copy(D)
    
    # Input cost matrix
    print("Enter the cost matrix (row-wise, comma-separated):")
    C = []
    for _ in range(m):
        row = [int(x) for x in input().split(",")]
        C.append(row)
    C = np.array(C)

    # Check for balanced problem
    if not np.isclose(np.sum(S), np.sum(D), atol=1e-10):
        print("The problem is not balanced!")
        return

    # Initialize variables
    x_nw = np.zeros((m, n))  # North-West corner method
    x_vogel = np.zeros((m, n))  # Vogel's approximation method
    x_russell = np.zeros((m, n))  # Russell's approximation method

    # North-West corner method
    i, j = 0, 0
    while i < m and j < n:
        quantity = min(S[i], D[j])
        x_nw[i, j] = quantity
        if quantity == S[i]:
            i += 1
        if quantity == D[j]:
            j += 1

    # Vogel's approximation method
    for i in range(m):
        for j in range(n):
            if x_vogel[i, j] == 0:
                mins = np.partition(C[i, :], 2)[:2]
                min_diff_row = np.abs(mins[0] - mins[1])
                max_diff_row = np.argmax(np.abs(C[:, j] - C[i, j]))

                mins = np.partition(C[:, j], 2)[:2]
                min_diff_col = np.abs(mins[0] - mins[1])
                max_diff_col = np.argmax(np.abs(C[i, :] - C[i, j]))

                if min_diff_row <= min_diff_col:
                    x_vogel[i, j] = min(S_copy[i], D_copy[j])
                    if x_vogel[i, j] == S_copy[i]:
                        i += 1
                    D_copy[j] -= x_vogel[i, j]
                else:
                    x_vogel[max_diff_col, j] = min(S_copy[max_diff_col], D_copy[j])
                    if x_vogel[max_diff_col, j] == D_copy[j]:
                        j += 1
                    S_copy[max_diff_col] -= x_vogel[max_diff_col, j]
                    D_copy[j] -= x_vogel[max_diff_col, j]

    # Russell's approximation method
    u = np.zeros(m)
    v = np.zeros(n)
    basic_cells = []

    while len(basic_cells) < m + n - 1:
        mins = []
        for i in range(m):
            for j in range(n):
                if x_russell[i, j] == 0:
                    mins.append((i, j, C[i, j] - u[i] - v[j]))

        if mins:
            i, j, _ = min(mins, key=lambda x: x[2])
            basic_cells.append((i, j))
            x_russell[i, j] = min(S_copy2[i], D_copy2[j])
            if x_russell[i, j] == S_copy2[i]:
                i += 1
            D_copy2[j] -= x_russell[i, j]
            if x_russell[i, j] == D_copy2[j]:
                j += 1
            if i < m:
                u[i] = C[i, j] - v[j]
            if j < n:
                v[j] = C[i, j] - u[i]
        else:
            break

    # Print output
    print("\nOutput:")
    print("Input Parameter Table:")
    print("Supply (S):", S)
    print("Demand (D):", D)
    print("Cost Matrix (C):")
    print(C)

    print("\nInitial Basic Feasible Solutions:")
    print("North-West Corner Method:")
    print(x_nw)
    print("\nVogel's Approximation Method:")
    print(x_vogel)
    print("\nRussell's Approximation Method:")
    print(x_russell)

# Call the function to execute the program
transportation_problem()
