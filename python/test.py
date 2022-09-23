from numpy import array


if __name__ == "__main__":
    my_list = [1]
    my_array = array(my_list)
    print(my_list)
    print(my_array)

    for list_entry in my_list:
        print(list_entry)

    for array_entry in my_array:
        print(array_entry)
