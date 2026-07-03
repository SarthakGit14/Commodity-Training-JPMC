import pandas as pd
import numpy as np

# 1. Load the data
df = pd.read_csv('Task 3 and 4_Loan_Data.csv')

def quantize_fico_log_likelihood(df, num_buckets=10):
    # Group data by distinct FICO scores for faster processing
    agg_df = df.groupby('fico_score').agg(
        n=('default', 'count'),
        k=('default', 'sum')
    ).reset_index().sort_values('fico_score')

    ficos = agg_df['fico_score'].values
    n_arr = agg_df['n'].values
    k_arr = agg_df['k'].values
    N = len(ficos)

    # Precompute Log-Likelihood for every possible continuous interval [i, j]
    LL = np.zeros((N, N))
    for i in range(N):
        n_sum, k_sum = 0, 0
        for j in range(i, N):
            n_sum += n_arr[j]
            k_sum += k_arr[j]
            if n_sum == 0:
                LL[i, j] = 0
            else:
                p = k_sum / n_sum
                ll = 0
                if k_sum > 0:
                    ll += k_sum * np.log(p)
                if (n_sum - k_sum) > 0:
                    ll += (n_sum - k_sum) * np.log(1 - p)
                LL[i, j] = ll

    # DP Initialization
    # dp[b][i] = max log-likelihood for the first i elements using b buckets
    dp = np.full((num_buckets + 1, N + 1), -np.inf)
    dp[0][0] = 0.0
    
    # Track the split index to reconstruct the boundaries later
    split = np.zeros((num_buckets + 1, N + 1), dtype=int)

    # Fill DP table
    for b in range(1, num_buckets + 1):
        for i in range(b, N + 1):
            # Try all possible split points 'm' for the last bucket
            for m in range(b - 1, i):
                val = dp[b - 1][m] + LL[m, i - 1]
                if val > dp[b][i]:
                    dp[b][i] = val
                    split[b][i] = m

    # Reconstruct optimal bucket start points
    boundaries = []
    curr = N
    for b in range(num_buckets, 0, -1):
        m = split[b][curr]
        if m < N:
            boundaries.append(ficos[m])
        curr = m
        
    boundaries.reverse()
    return boundaries

# 2. Execute Quantization
num_buckets = 10
optimal_boundaries = quantize_fico_log_likelihood(df, num_buckets)

print("Optimized FICO Bucket Start Boundaries:")
print(optimal_boundaries)

# 3. Create Rating Mapping Function
def assign_rating(fico, boundaries):
    """
    Maps a FICO score to a Rating.
    Lower rating index signifies a better credit score (Rating 1 = Best).
    """
    # Iterate backwards so higher FICO scores get lower ratings (e.g., Rating 1)
    for i in range(len(boundaries) - 1, -1, -1):
        if fico >= boundaries[i]:
            return len(boundaries) - i
    return len(boundaries)

# Apply mapping to the dataset
df['rating'] = df['fico_score'].apply(lambda x: assign_rating(x, optimal_boundaries))

# 4. Analyze Results
summary = df.groupby('rating').agg(
    min_fico=('fico_score', 'min'),
    max_fico=('fico_score', 'max'),
    total_borrowers=('default', 'count'),
    defaults=('default', 'sum')
).reset_index()

summary['probability_of_default'] = summary['defaults'] / summary['total_borrowers']
print("\n--- Final Rating Map Summary ---")
print(summary.round({'probability_of_default': 4}).to_string(index=False))