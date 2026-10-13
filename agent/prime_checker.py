def is_prime(n):
    """Return True if n is a prime number, False otherwise."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def main():
    try:
        num_str = input("Enter an integer to check if it is prime: ")
        num = int(num_str)
    except ValueError:
        print("Invalid input. Please enter a valid integer.")
        return

    if is_prime(num):
        print(f"{num} is a prime number.")
    else:
        print(f"{num} is not a prime number.")


if __name__ == "__main__":
    main()
