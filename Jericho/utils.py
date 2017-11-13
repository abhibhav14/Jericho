def compute_gini(agent_wealths):
    if sum(agent_wealths) == 0:
        return 0
    x = sorted(agent_wealths)
    N = len(x)
    B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return (1 + (1/N) - 2*B)
