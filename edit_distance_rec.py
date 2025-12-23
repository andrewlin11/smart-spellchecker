def distance_fast(s, t):
    """
    Calculate Levenshtein distance using dynamic programming.
    Much faster than recursive approach.
    """
    m, n = len(s), len(t)
    
    if m == 0:
        return n
    if n == 0:
        return m
    
    # Initialize first row
    prev = list(range(n + 1))
    
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        
        for j in range(1, n + 1):
            # Cost of substitution
            cost = 0 if s[i-1] == t[j-1] else 1
            
            # Take minimum of insert, delete, substitute
            curr[j] = min(
                curr[j-1] + 1,      # Insert
                prev[j] + 1,        # Delete
                prev[j-1] + cost    # Substitute
            )
        
        prev = curr
    
    return prev[n]

# Keep old recursive version for compatibility
cost_sub = 2
cost_ins = 1
cost_del = 1

def distance(s, t):
    if len(s) == 0: return cost_del*len(t)
    elif len(t) == 0: return cost_ins*len(s)
    cost = cost_sub
    if s[-1] == t[-1]:
        cost = 0
    return min( distance(s[:-1], t) + cost_ins,
                distance(s, t[:-1]) + cost_del,
                distance(s[:-1], t[:-1]) + cost)


def main():
    print("Distance between sprite and prize:")
    print(distance("sprie", "prize"))
    print(distance("eradication", "elucidation"))

if __name__ == '__main__':
    main()
