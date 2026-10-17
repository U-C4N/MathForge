"""The independent checker of the demo v6 family: it rechecks each value without importing the producer."""


def check_operation(operation, args, value, evidence):
    if evidence != ():
        return False
    if operation == "demo_double":
        n = args.get("n")
        return type(n) is int and type(value) is int and value == n + n
    if operation == "demo_echo":
        return value == args["expression"]
    if operation == "demo_rows":
        return type(value) is int and value == len(args["A"].rows)
    return False
