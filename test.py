import random

def display(matrix):
    for i in matrix:
        print(i)

class RBAllocator:
    def __init__(self, K, B, M):
        self.K = K  # number of users (rows)
        self.B = B  # number of RBs (columns)
        self.M = M  # upper bound for random values (1..M-1)
        
        # Initialize the matrix: K x B, with each column having exactly one 1 (rest zeros)
        self.matrix = [[0 for _ in range(B)] for _ in range(K)]
       
        if K == B:
            # Create a permutation matrix (each row and column has exactly one 1)
            cols = random.sample(range(B), B)
            for row in range(K):
                self.matrix[row][cols[row]] = 1
        elif K < B:
            # Ensure each row has at least one 1, and each column has exactly one 1
            # First, assign one 1 to each row to satisfy the row constraint
            cols = random.sample(range(B), K)  # Select K columns randomly
            for row in range(K):
                self.matrix[row][cols[row]] = 1
            # Assign remaining B-K columns to rows randomly
            remaining_cols = [j for j in range(B) if j not in cols]
            for col in remaining_cols:
                row = random.randint(0, K-1)
                self.matrix[row][col] = 1
        else:  # K > B
            # Ensure each row has at most one 1, and each column has exactly one 1
            # Randomly assign each column to a unique row
            rows = random.sample(range(K), B)  # Select B rows randomly
            for col in range(B):
                self.matrix[rows[col]][col] = 1
        
        print("Initial matrix:")
        for row in self.matrix:
            print(row)
        self.current_pos = (0, 0)  # (row, column), starts at (0,0)
    
    def update_matrix(self):
        rows = self.K
        cols = self.B
        i, j = self.current_pos
        
        found = False
        for _ in range(rows * cols):
            # Check current position
            if self.matrix[i][j] != 0:
                # Replace with random value in [1, M-1]
                new_val = random.randint(1, self.M - 1)
                self.matrix[i][j] = new_val
                found = True
            
            # Move to next position (row-major order)
            j += 1
            if j >= cols:
                j = 0
                i += 1
                if i >= rows:
                    i = 0
            
            if found:
                break
        
        # Update the next starting position
        self.current_pos = (i, j)
        
        return self.matrix
    
allocator = RBAllocator(5, 4, 7)
print()
display(allocator.matrix)