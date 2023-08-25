# Step 1: Load Data
# load cheaters data 
with open('../assignment-final-data/cheaters.txt') as c:
    ch = c.readlines()
    cheaters = [i.split('\t', 1)[0] for i in ch]
# I should modularise this to a function later for extra marks.  

# Load teams data
with open('../assignment-final-data/team_ids.txt') as t:
    tm = t.readlines()
    teams = [i.split('\t') for i in tm]
# I should modularise this to a function later for extra marks.  

# Step 2: create dummy variable that indicates whether a player is a cheater or not
for i in range(len(teams)):
    if teams[i][1] in cheaters:
        teams[i].append(1)
    else:
        teams[i].append(0)
        
# Step 3: create a dictionary that counts the number of cheaters in each team
match_team_cheaters_dic = {}

for list_in_list in teams:
    match_id, _, team_id, cheater_dummy = list_in_lists
    key = (match_id, team_id)
    if key in match_team_cheaters_dic:
        match_team_cheaters_dic[key] += cheater_dummy
    else:
        match_team_cheaters_dic[key] = cheater_dummy
        
actual_cheaters_list = []
for i in 0,1,2,3,4:
    actual_cheaters_list.append(list(match_team_cheaters_dic.values()).count(i))
    
# Step 4: Print out the number of teams with X number of cheaters.
print("There are", actual_cheaters_list[0],"instances of 0 cheaters in a team.")
print("There are", actual_cheaters_list[1],"instances of 1 cheater in a team.")
print("There are", actual_cheaters_list[2],"instances of 2 cheaters in a team.")
print("There are", actual_cheaters_list[3],"instances of 3 cheaters in a team.")
print("There are", actual_cheaters_list[4],"instances of 4 cheaters in a team.")

# Step 5: Create list with match_id and team_id
ls = []
for i in range(len(teams)):
    ls.append(teams[i][0:3:2])

# Step 6: Create dictionary with match_id as key and team_ids as values. 
match_team_ids = {i[0]:[] for i in ls}

for i, j in ls:
    match_team_ids[i].append(j)

# Step 7: Shuffle the values in the dictionary
# function to randomly shuffle the values in a dictionary per key 
def shuffle_dict_values(dic): 
    '''Function takes dictionary as argument and returns a dictionary with the values for each key randomly shuffled.'''
	# iterate over each key in the dictionary 
	for key in dic: 

		# randomly shuffle the values for each key 
		random.shuffle(dic[key]) 

	return dic


# Step 8: Get key-value pairs in list of lists format

def get_key_value_pairs(dictionary):
  result = []
  for key, values in dictionary.items():
    for value in values:
      result.append([key, value])
  return result

# Step 9: Add player_id and cheater_dummy back into key/value pairs (now shuffled)

for i in range(len(match_team_pairs)):
    match_team_pairs[i].append(teams[i][3])
    match_team_pairs[i].append(teams[i][1])

new_teams = match_team_pairs

# Step 10: Convert to dictionary that stores number of cheaters in team per match_id/team_id pair

cheaters_random_dic = {}

for list_in_lists in new_teams:
    match_id, team_id, cheater_dummy, _, = list_in_lists
    key = (match_id, team_id)
    if key in cheaters_random_dic:
        cheaters_random_dic[key] += cheater_dummy
    else:
        cheaters_random_dic[key] = cheater_dummy
        
        
# Step 11: Count total number of teams with 0,1,2,3,4 cheaters, and store in list
cheaters_count_list = []
for i in 0,1,2,3,4:
    cheaters_count_list.append(list(cheaters_random_dic.values()).count(i))

# Step 12: Print out values 

print("There are", cheaters_count_list[0],"instances of 0 cheaters in a team.")
print("There are", cheaters_count_list[1],"instances of 1 cheater in a team.")
print("There are", cheaters_count_list[2],"instances of 2 cheaters in a team.")
print("There are", cheaters_count_list[3],"instances of 3 cheaters in a team.")
print("There are", cheaters_count_list[4],"instances of 4 cheaters in a team.")