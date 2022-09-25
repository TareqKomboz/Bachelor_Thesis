from numpy import array


if __name__ == "__main__":
    a = array([0.0, True, "hallo"])
    b = array([0.0])
    c = array([0, 20, 3])
    for entry in a:
        print(type(entry))
    for entry in b:
        print(type(entry))
    for entry in c:
        print(type(entry))
