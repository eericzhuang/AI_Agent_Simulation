def fibonacci(n):
    """Return the Fibonacci sequence up to the nth term."""
    if n <= 0:
        return []
    elif n == 1:
        return [0]

    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i - 1] + sequence[i - 2])
    return sequence


if __name__ == "__main__":
    n = 10
    result = fibonacci(n)
    print(f"Fibonacci sequence ({n} terms): {result}")
