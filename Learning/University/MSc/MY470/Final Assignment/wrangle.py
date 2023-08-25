'''This module contains functions for transforming, enriching, wrangling and creating new derivative data.'''

from datetime import datetime
import random
import calculate
import load
import statistics as stats


def add_dummy_end(list_of_lists, helper_list):
    ''' This function takes a list of lists and a helper list and adds a dummy variable to each list in the list of lists with
    a value of 1 if the second element is in the helper list and 0 if not..'''
    for i in range(len(list_of_lists)):
        # if second element in list is in helper list, add 1 to list
        if list_of_lists[i][1] in helper_list:
            list_of_lists[i].append(1)
        else:
            list_of_lists[i].append(0)


def get_cheaters_per_match_team(list_of_lists):
    '''This function takes a list of 5 lists as an argument and returns 5 separate lists with the values of each list in 
    each individual list.'''
    zero_cheaters = []
    one_cheater = []
    two_cheaters = []
    three_cheaters = []
    four_cheaters = []

    for sublist in list_of_lists:
        for sub_sublist in sublist:
            zero_cheaters.append(sub_sublist[0])
            one_cheater.append(sub_sublist[1])
            two_cheaters.append(sub_sublist[2])
            three_cheaters.append(sub_sublist[3])
            four_cheaters.append(sub_sublist[4])

    return zero_cheaters, one_cheater, two_cheaters, three_cheaters, four_cheaters


def get_dict(list_of_list):
    '''This function takes a list of lists, with each list having two elements and creates a dictionary that maps the first element as the key
    and the second element in the list as the associated value.'''
    cheat_date_dic = {player_id: date for player_id, date in list_of_list}
    return cheat_date_dic


def add_dummy(list_of_lists, dictionary):
    '''This function takes a list of lists and a dictionary with 1 value per key and add 
    a dummy variable to each list in the list of lists'''
    kill_cheat = []

    for k in list_of_lists:
    # Unpack kill for legibility
        match_id, killer_id, victim_id, kill_time = k
        is_killer_cheater_at_time = 0 
    # Establish whether killer was a cheater during time of kill and add dummy 
        if killer_id in dictionary and dictionary[killer_id] <= kill_time:
                is_killer_cheater_at_time = 1 

        kill_cheat.append([match_id, killer_id, victim_id, kill_time, is_killer_cheater_at_time])

    return kill_cheat

def add_dummy_2(list_of_lists, dictionary):
    '''This function takes a list of lists and a dictionary as an argument and adds a dummy variable to each list in the list of lists
    with 1 if the first element in each list is in the dictionary and the value of the 
    dictionary is greater than the second element in the list, and 0 if not.'''
    observers_became_cheaters = []

    for k in list_of_lists:
    # Unpack kill for legibility
        victim_id, kill_time = k
        observer_become_cheater_after = 0
    # Establish whether observer became a cheater after time of kill
        if victim_id in dictionary and dictionary[victim_id] > kill_time:
        
            observer_become_cheater_after = 1

        observers_became_cheaters.append([victim_id, kill_time, observer_become_cheater_after])

    return observers_became_cheaters


def add_dummies(list_of_lists, dictionary):
  '''This function takes a list of lists and a dictionary with 1 value per key and adds 
  3 dummy variables to each list in the list of lists'''
  
  kill_cheat = []

  for k in list_of_lists:
  # Unpack kill for legibility
        match_id, killer_id, victim_id, kill_time = k
        is_killer_cheater_at_time = 0 
        is_victim_cheater = 0
        vict_become_cheater_after = 0
  # First, establish whether killer was a cheater during time of kill
        if killer_id in dictionary and dictionary[killer_id] <= kill_time:
          is_killer_cheater_at_time = 1 
  # Second, establish whether victim became a cheater at any point AFTER being killed
        if victim_id in dictionary and dictionary[victim_id] > kill_time:
          is_victim_cheater = 1   
  # Third, establish whether victim became a cheater at any point AFTER being killed by a cheater
        if is_killer_cheater_at_time == 1 and is_victim_cheater == 1:
          vict_become_cheater_after = 1

        kill_cheat.append([match_id, killer_id, victim_id, kill_time, is_killer_cheater_at_time, 
                    is_victim_cheater, vict_become_cheater_after])

  return kill_cheat


def create_list_all_values(dictionary):
    '''Takes a dictionary as an argument and returns a list of all unique values in the dictionary.'''
    all_players = []
    for list in dictionary.values():
        for i in list:
            all_players.append(i)

    all_players_3 = set(all_players)
    return all_players_3


def get_dict_mult_values(list_of_lists):
    '''This function takes a list of lists and returns a dictionary with the first element of each sublist as the key and the second element of each sublist as the value. 
    If there are multiple values for the same key, they are stored in a list.'''
    ls = []

    for i in range(len(list_of_lists)):
        # append ls with the first and second element of each sublist
        ls.append(list_of_lists[i][0:2:1])

    match_killer_ids = {i[0]:[] for i in ls}

    for i, j in ls:
        match_killer_ids[i].append(j)
    
    return match_killer_ids


def get_dict_mult_values_2(list_of_lists):
    '''This function takes a list of lists and returns a dictionary with the first element of each sublist as the key and the third element of each sublist as the value.
    If there are multiple values for the same key, they are stored in a list.'''
    ls = []

    for i in range(len(list_of_lists)):
        # access the first and third element of each sublist
        ls.append(list_of_lists[i][0:3:2])

    match_victim_ids = {i[0]:[] for i in ls}

    for i, j in ls:
        match_victim_ids[i].append(j)

    return match_victim_ids


def get_dict_unique_totals_random(keys, dict_1, dict_2):
    '''Function takes two dictionaries with the same keys and returns two dictionary with the same keys and 
    the unique value entries across both dictionaries per key. The second dictionary will have the values per key
    in random order.'''
    total_players = {}
      
    for k in keys:
        total_players[k] = list(set(dict_1[k] + dict_2[k]))
          
    total_players_ref = {}
      
    for k in keys:
        total_players_ref[k] = list(set(dict_1[k] + dict_2[k]))
        random.shuffle(total_players_ref[k])
    
    return total_players, total_players_ref


def swap_by_index_2_elements(list_of_lists, dict_1, dict_1_randomised, dict_2, dict_3):
    '''This function takes a list of lists, two dictionaries with the same keys, but the second dictionary has the values in random order, 
    and two dictionaries with the same keys. The function returns a list of lists with the first element of each sublist 
    as the first element of each sublist in the original list of lists, the second and third element from the randomised dictionary with index corresponding to
    the non-randomised dictionary and the fourth element as the fourth element of each sublist in the original list of lists..'''

    new_killers = []
    new_victims = []
    lol = list_of_lists

    for match, values in dict_1.items():
        # for each value in dict_2, find the index of the value in dict_1 and use that 
        # index to find the corresponding value in dict_1_randomised
        for killer in dict_2[match]:
            ind = dict_1[match].index(killer)
            new_killers.append(dict_1_randomised[match][ind])
        # for each value in dict_3, find the index of the value in dict_1 and use that
        # index to find the corresponding value in dict_1_randomised
        for victim in dict_3[match]:
            ind = dict_1[match].index(victim)
            new_victims.append(dict_1_randomised[match][ind])

    rand_kills = []

    for i in range(len(lol)):
        # append returned list with the first element of each sublist in the original list of lists, 
        # the second and third element from the randomised dictionary with index corresponding to
        # the non-randomised dictionary and the fourth element as the fourth element of each sublist in the original list of lists.
        rand_kills.append([lol[i][0], new_killers[i], new_victims[i], lol[i][3]])
    return rand_kills 


def make_complete_dictionary(list_of_lists):
    """This function takes a list of lists, groups by the first element of each list as the key and populates 
    the values with the other elements of each list with the same first element. 
    The function returns a dictionary with the first element of each list as the key and the other elements of each list
    with the same first element as the values."""
    all_kills_per_match = {}
    for list in list_of_lists:
        match_id, killer_id, victim_id, kill_time, cheater_dummy = list
        # if the key is not in the dictionary, add it and append an empty value
        if match_id not in all_kills_per_match:
            all_kills_per_match[match_id] = []
        # append the values to the key
        all_kills_per_match[match_id].append([killer_id, victim_id, kill_time, cheater_dummy])
    return all_kills_per_match


def make_complete_dictionary_2(list_of_lists):
    """This function takes a list of lists, groups by the first element of each list as the key and populates the values with the other elements of each list with the same
    first element. The function returns a dictionary with the first element of each list as the key and the other elements of each list with the same first element as the
    values."""
    all_kills_per_match = {}
    for list in list_of_lists:
        match_id, killer_id, victim_id, kill_time, cheater_dummy = list
        if match_id not in all_kills_per_match:
            all_kills_per_match[match_id] = []
        all_kills_per_match[match_id].append([killer_id, victim_id, kill_time, cheater_dummy])
    return all_kills_per_match


def get_unique_dict_values(dictionary):
    '''This function takes a dictionary and returns a dictionary with all the unique values now as key.'''
    all_players = []
    for list in dictionary.values():
        for i in list:
            all_players.append(i)

    all_players_3 = set(all_players)
    
    players_dict = {i: None for i in all_players_3}
    return players_dict


def extract_entries_per_key_after_point(dictionary, stop_point):
    '''This function takes a dictionary and a numeric value as stop_point and returns a list
    with all the entries per key after the stop_point where the fourth element in each list is equal to 1.'''
    kills_after = [] # list of all observers that I will add to and check if they were cheaters 

    for match in dictionary: # looking at each match
        helper_dict = {} # empty dictionary for each match, for cheaters and kill count to be added to
        # i is the index of the kill in the match
        for i in range(len(dictionary[match])): 
            # now, just look at those who were cheater at the time of each match
            if dictionary[match][i][3] == 1:
                # if the cheater is not already in the helper dictionary as a key
                if dictionary[match][i][0] not in helper_dict: 
                     # initialise the cheater in the helper dictionary with 1 kill
                    helper_dict[dictionary[match][i][0]] = 1
                                
                else: 
                    # add 1 to the number of kills
                    helper_dict[dictionary[match][i][0]] += 1 
                # if the cheater has stop_point number of kills                
                if helper_dict[dictionary[match][i][0]] == stop_point: # if the cheater has 3 kills
                    # add the rest of the kills in the match to the observers list
                    kills_after.append([dictionary[match][i + 1:]]) 
    return kills_after


def extract_2nd_3rd_element_lolol(list_of_lists_of_lists):
    '''This function takes a list of lists of list, with each sublist containing a list of lists 
    and returns a list of lists with the second and third element of each sublist in the list of lists.'''
    observers = []
    # for each list of lists in the list of lists of lists
    for i in range(len(list_of_lists_of_lists)):
        # for each sublist in the list of lists
        for j in range(len(list_of_lists_of_lists[i][0])):
            # append the second and third element of each sublist in the list of lists
            observers.append([list_of_lists_of_lists[i][0][j][1], list_of_lists_of_lists[i][0][j][2]])
    return observers


def create_dict_conditional_value(list_of_lists, numerical_condition):
    '''This function takes a list of lists and a numerical condition and returns a dictionary of 
    the first element of the list of lists as the key  and the second element of the list of lists 
    as the value if the second element of the list of lists is equal to the numerical condition.'''
    return {i[0]:i[2] for i in list_of_lists if i[2] == numerical_condition}
