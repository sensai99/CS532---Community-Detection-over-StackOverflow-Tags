import numpy as np

def cosine_similarity(vec1, vec2):
    # Convert the input lists or arrays to NumPy arrays
    vec1, vec2 = np.array(vec1), np.array(vec2)
    # Compute the dot product of the two vectors
    dot_product = np.dot(vec1, vec2)
    # Calculate the Euclidean norm (or L2 norm) of each vector
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    # Calculate cosine similarity by dividing the dot product by the product of the norms
    # Return as float and preventing division by zero
    return float(dot_product / (norm_a * norm_b)) if norm_a != 0 and norm_b != 0 else 0.0