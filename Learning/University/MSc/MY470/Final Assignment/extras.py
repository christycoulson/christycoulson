# Randomisation for part 2 
def get_random_v2c_2(list_of_lists):
    '''Fill in this docstring'''
    
    lol = list_of_lists
     # 1. Dictionary with Match_id & killer_id
    ls = []

    for i in range(len(lol)):
        ls.append(lol[i][0:2:1])

    match_killer_ids = {i[0]:[] for i in ls}

    for i, j in ls:
        match_killer_ids[i].append(j)
        
    # 2. Dictionary with Match_id & victim_id

    ls = []

    for i in range(len(lol)):
        ls.append(lol[i][0:3:2])

    match_victim_ids = {i[0]:[] for i in ls}

    for i, j in ls:
        match_victim_ids[i].append(j)
        
    match_ids = match_killer_ids.keys()  
    total_players = {}
    total_players_ref = {}
    
    for k in match_ids:
        total_players[k] = list(set(match_killer_ids[k] + match_victim_ids[k]))

    for k in match_ids:
        total_players_ref[k] = list(set(match_killer_ids[k] + match_victim_ids[k]))
        random.shuffle(total_players_ref[k])
    
    new_killers = []
    new_victims = []

    for match, values in total_players.items():
    
        for killer in match_killer_ids[match]:
            ind = total_players[match].index(killer)
            new_killers.append(total_players_ref[match][ind])
            
        for victim in match_victim_ids[match]:
            ind = total_players[match].index(victim)
            new_victims.append(total_players_ref[match][ind])
 
    rand_kills = []

    for i in range(len(list_of_lists)):
        rand_kills.append([lol[i][0], new_killers[i], new_victims[i], lol[i][3]])
        
      # 2. Create dummy variable for if killer was a cheater at the time of the kill
    rand_kill_cheat = []

    for k in rand_kills:
        # Unpack kill for legibility
        match_id, killer_id, victim_id, kill_time = k
        is_killer_cheater_at_time = 0 
        is_victim_cheater = 0
        vict_become_cheater_after = 0
        # First, establish whether killer was a cheater during time of kill
        if killer_id in cheat_date_dic and cheat_date_dic[killer_id] <= kill_time:
            is_killer_cheater_at_time = 1 
        # Second, establish whether victim became a cheater at any point AFTER being killed
        if victim_id in cheat_date_dic and cheat_date_dic[victim_id] > kill_time:
            is_victim_cheater = 1   
        # Third, establish whether victim became a cheater at any point AFTER being killed by a cheater
        if is_killer_cheater_at_time == 1 and is_victim_cheater == 1:
            vict_become_cheater_after = 1

        rand_kill_cheat.append([match_id, killer_id, victim_id, kill_time, is_killer_cheater_at_time, 
                        is_victim_cheater, vict_become_cheater_after])
        
    part_2_random_count = (sum(i[6] == 1 for i in rand_kill_cheat))
    
    print(part_2_random_count)
    
    return part_2_random_count


