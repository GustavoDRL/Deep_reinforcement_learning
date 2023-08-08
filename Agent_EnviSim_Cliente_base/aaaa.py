import numpy as np

def create_probabilistic_distribution_with_zeros_preserved(scores):
    # Step 1: Identify non-zero elements
    non_zero_mask = scores != 0

    # Step 2: Normalize non-zero elements' scores
    non_zero_scores = scores[non_zero_mask]
    max_non_zero_score = np.max(non_zero_scores)
    exp_non_zero_scores = np.exp(non_zero_scores - max_non_zero_score)

    # Step 3: Calculate the sum of all exponential scores for non-zero elements
    sum_exp_non_zero_scores = np.sum(exp_non_zero_scores)

    # Step 4: Compute the probabilities for each non-zero element
    probabilities = np.zeros_like(scores)
    probabilities[non_zero_mask] = exp_non_zero_scores / sum_exp_non_zero_scores

    return probabilities

# Example usage:
input_scores = np.array([2.0, 1.0, 0.1, 0.0, 0.0, 3.0])
prob_distribution = create_probabilistic_distribution_with_zeros_preserved(input_scores)
print(prob_distribution)
