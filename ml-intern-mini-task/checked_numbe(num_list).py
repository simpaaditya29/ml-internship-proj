def checked_number(num_list):
    for i in num_list:
        if i == 0:
            print("Zero found in the list.")
        elif i > 0:
            print("Positive number found in the list.")
        else:
            print("Negative number found in the list.")    

checked_number([1, -2, 3, 0, -5, 6])