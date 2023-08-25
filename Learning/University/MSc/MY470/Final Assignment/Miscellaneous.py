def group_by(rows, key):
    m = {}
    for row in rows:
        k = key(row)
        try:
            m[k].append(row)
        except KeyError:
            m[k] = [row]
    return m.values()

def in_list(list_of_lists, string):
    '''Takes a list of lists and a string as argument. Iterates through each list within the list of lists and returns 
    True if the string is found in any of the lists. Returns False if the string is not found in any of the lists. '''
    for i in list_of_lists:
        if string in i:
            return True
    return False