"""The independent checker of the demo v6 family: it recomputes demo_double without importing the producer."""


def check_operation(operation, args, value, evidence):
    n = args.get("n")
    return (operation == "demo_double" and evidence == () and type(n) is int and type(value) is int
            and value == n + n)
