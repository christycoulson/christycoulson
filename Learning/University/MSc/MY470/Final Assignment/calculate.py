'''This module contains all of the functions for the calculations required for the final project. This is separate from the wrangle file in that the functions
included here are not used to manipulate the data, but rather to calculate the statistics required for the final project.'''

import statistics 
import math
import random
import wrangle
import load

def count_1(list_of_lists, element):
    '''This function takes a list of a list and an element of each list in a list of lists and 
    counts the number of times an element's value is equal to 1 in a list of lists.'''
    return sum(i[element] == 1 for i in list_of_lists)

def get_stats_list(list):
    '''This function take a list of numerical values as an argument and returns a list of the 
    mean, standard deviation, upper confidence interval, and lower confidence interval'''
    expected_val = statistics.mean(list)
    sd = statistics.stdev(list)
    upper_ci = expected_val + 1.96 * sd / math.sqrt(len(list))
    lower_ci = expected_val - 1.96 * sd / math.sqrt(len(list))
    stats = [expected_val, sd, upper_ci, lower_ci]
    return stats

def calc_dummy_count_per_key(list_of_lists):
    '''This function takes a list of lists as an argument, with the fourth element in each list as numeric and 
    returns a dictionary with the first element in each list of lists as the key and the sum of the the fourth, numeric element
    as the value for each key.'''
    match_team_cheaters_dic = {}
    for list_in_lists in list_of_lists:
        # _, to ignore the second element in each list
        match_id, _, team_id, cheater_dummy = list_in_lists
        # identify key to populate dictionary wtih
        key = (match_id, team_id)
        # If the key is already in the dictionary, add the value of the fourth element to the value of the key
        if key in match_team_cheaters_dic:
            match_team_cheaters_dic[key] += cheater_dummy
        else:
            match_team_cheaters_dic[key] = cheater_dummy
    return match_team_cheaters_dic

def get_no_cheaters(dic, number):
    '''This function takes a dictionary with numeric values and a number as arguments and returns a list of the count of times
    each number, from 0 to the number argument, appears in the dictionary values.'''
    actual_cheaters_list = []
    # for each number in the range of the number of cheaters, count the number of times that number appears in the dictionary values
    for i in range(number):
        actual_cheaters_list.append(list(dic.values()).count(i))
    return actual_cheaters_list

def get_random_teams_per_cheaters(list_of_lists):
    '''Function takes a list of lists as input and returns a list of lists of lists.
     The function shuffles the values in each key and then calculates the number of keys with values of 0, 1, 2, 3 and 4 '''
    cheaters_per_team = []
    
    ls = []
    # create a list of lists with the first and third elements of each list in the list of lists
    for i in range(len(list_of_lists)):
        ls.append(list_of_lists[i][0:3:2])
        
    # create dictionary with first element of each list in ls as key and empty list as value
    match_team_ids = {i[0]:[] for i in ls}

    # populate dictionary with second element of each list in ls as value for each key
    for i, j in ls:
        match_team_ids[i].append(j)
        
    # shuffle the values for each key
    for key in match_team_ids: 
        # randomly shuffle the values for each key 
        random.shuffle(match_team_ids[key]) 

    result = []
    # create edge_pairs of match_id and team_id in 1-1 pairs
    for key, values in match_team_ids.items():
        for value in values:
            result.append([key, value])

    # function to add player_id and cheater_dummy back into shuffled teams list of lists
    for i in range(len(result)):
        result[i].append(list_of_lists[i][3])
        result[i].append(list_of_lists[i][1])

    # allocate to new_team alias
    new_teams = result

    cheaters_random_dic = {}
    for list_in_lists in new_teams:
    # _, to ignore the second element in each list   
        match_id, team_id, cheater_dummy, _, = list_in_lists
    # identify key to populate dictionary wtih
        key = (match_id, team_id)
    # If the key is already in the dictionary, add the value of the fourth element to the value of the key
        if key in cheaters_random_dic:
            cheaters_random_dic[key] += cheater_dummy
        else:
            cheaters_random_dic[key] = cheater_dummy

    cheaters_count_list = []
    for i in 0,1,2,3,4:
        cheaters_count_list.append(list(cheaters_random_dic.values()).count(i))
    
    cheaters_per_team.append(cheaters_count_list)
    
    return cheaters_per_team


def count_obs_per_1st_element(list_of_list):
    '''Takes a list of list as an argument and returns a dictionary with the first element in each list as the key 
    and the count of the number of times that key appears in the list of lists as the value.'''
    # Create an empty dictionary to store the results
    kills_per_match = {}

    # Iterate through the list of kills
    for list in list_of_list:
        # Get the match ID from the list
        match_id = list[0]  
        # Get the killer ID from the list
        killer_id = list[1]

        # Check if we already have an entry for this match and killer in our dictionary
        if (match_id, killer_id) in kills_per_match:
            # If so, increment the count by 1
            kills_per_match[(match_id, killer_id)] += 1

        else: 
            # Otherwise, create a new entry with a count of 1 
            kills_per_match[(match_id, killer_id)] = 1

    return kills_per_match