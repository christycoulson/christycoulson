'''This module contains functions for reading and manipulating data from files.'''

from datetime import datetime 
import calculate 
import wrangle

def load_all_sorted(string):
    '''This function takes a string (file pathway) as an argument and returns a list of lists of the data in the file, 
    sorted by the first element in each list.'''
    with open(string) as t:
        tm = t.readlines()
        data = [i.strip().split('\t') for i in tm]
        data_sorted = sorted(data, key=lambda x: x[0])
    return data_sorted

def load_all_sorted_match(string):
    '''This function takes a string (file pathway) as an argument and returns a list of lists of the data in the file,
    sorted by the first element in each list. The fourth element in each list is converted to a datetime object.'''
    with open(string) as t:
        tm = t.readlines()
        data = [i.strip().split('\t') for i in tm]
        data_sorted = sorted(data, key=lambda x: x[0])
        for i in range(len(data_sorted)):
            data_sorted[i][3] = datetime.strptime(data_sorted[i][3], '%Y-%m-%d %H:%M:%S.%f').date() 
        return data_sorted
    
def load_all_sorted_match_date(string):
    '''This function takes a string (file pathway) as an argument and returns a list of lists of the data in the file,
    sorted by the first and second element in each list. The fourth element in each list is converted to a datetime object.'''
    with open(string) as t:
        tm = t.readlines()
        data = [i.strip().split('\t') for i in tm]
        data_sorted = sorted(data, key=lambda x: (x[0], x[3]))
        for i in range(len(data_sorted)):
            data_sorted[i][3] = datetime.strptime(data_sorted[i][3], '%Y-%m-%d %H:%M:%S.%f').date() 
        return data_sorted

def load_first_line(string):
    '''This function takes a string (file pathway) as an argument and returns a list of the first element 
    in each list of the data in the file.'''
    with open(string) as c:
        ch = c.readlines()
        data_first = [i.strip().split('\t', 1)[0] for i in ch]
    return data_first


def load_first_two_lines(string):
    '''This function takes a string (file pathway) as an argument and returns a list of lists of the first two elements
    in each list of the data in the file. The second element in each list is converted to a datetime object.'''
    with open(string) as c:
        ch = c.readlines()
        data_first_2 = [i.strip().split('\t')[0:2] for i in ch]
        for i in range(len(data_first_2)):
            data_first_2[i][1] = datetime.strptime(data_first_2[i][1], '%Y-%m-%d').date() 
    return data_first_2


