def is_palindrome(n: int) -> bool:
    """Return True if *n* is a palindrome number, False otherwise.

    Parameters
    ----------
    n : int
        The number to test.

    Returns
    -------
    bool
        True if *n* is a palindrome, False otherwise.
    """
    s = str(n)
    return s == s[::-1]


if __name__ == "__main__":
    for num in [121, 123, -121, 0, 11, 10]:
        print(f"{num}: {is_palindrome(num)}")