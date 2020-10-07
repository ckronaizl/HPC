#all advanced calculations come from the following sources:
# https://www.nbastuffer.com/analytics101/offensive-efficiency/

import csv
import os

def ts_calc(team):
    #tmPTS = team[8]
    #tmFGA = team[10]
    #tmFTA = team[16]
    if team[16] == "0" or team[16] == "-" or team[16] == 0:
        return 0
    else:
        return str(100*int(team[8])/(2*(int(team[10]) + (0.44*int(team[16])))))

def oer_calc(team):
    #tmPOSS = team[27]
    #tmPTS = team[8]
    if len(team) > 27:
        if float(team[27]) > 0:
            return str(100*int(team[8])/float(team[27]))
    return 0

def possessions_calc(team, opponent):
    possessions = 0
    #tmFGA = team[10]
    #tmFTA = team[16]
    #tmFTM = team[15]
    #tmORB = team[18]
    #opDRB = opponent[19]
    #tmFG = team[9]
    #tmTO = team[24]

    #tm_ORB_PCT = team[18]/(team[18] + opponent[19])
    #tm_SCORE_POSS = team[9] + (1-(1-(team[15]/team[16]))**2) * team[16] * 0.44
    #tm_PLAY_PCT = tm_SCORE_POSS / (team[10] + team[16] * 0.4 + team[24])
    #tm_ORB_WGT = ((1 - tm_ORB_PCT) * tm_PLAY_PCT) / ((1-tm_ORB_PCT) * tm_PLAY_PCT + tm_ORB_PCT * (1-tm_PLAY_PCT))
    #ORB_PART = team[18] * tm_ORB_WGT * tm_PLAY_PCT

    try:
        possessions = (int(team[10])+0.4*int(team[16])-1.07*(int(team[18])/(int(team[18])+int(opponent[19])))*(int(team[10])-int(team[9]))+int(team[24]))
        possessions = 0.5*(possessions+(int(opponent[10])+0.4*int(opponent[16])-1.07*(int(opponent[18])/(int(opponent[18])+int(team[19])))*(int(opponent[10])-int(opponent[9]))+int(opponent[24])))
    except:
        return "0"
    return str(possessions)

#begin main
team_avg = []

for stats_file in os.listdir("C:/local/nba_stats/"):

    if stats_file.find("teams") > -1 and stats_file.find("adv") == -1 and stats_file.find("avg") == -1:
        with open("C:/local/nba_stats/" + stats_file,'r') as seasonstats:
            rawstats = seasonstats.readlines()

        stats = []
        games = 0
        for line in rawstats:
            line = line[:len(line)-1]
            stats.append(line.split(','))

        team_avg.clear()

        for x in range(1,len(stats)):
            #stats[x][26] = stats[x][26][:len(stats[x][26])-1]
            if len(stats[x]) < 28:

                for y in range(x,len(stats)):

                    #find team matchup
                    if stats[x][1] == stats[y][5] and stats[x][2] == stats[y][2] and len(stats[x]) < 30:

                        stats[x].append(possessions_calc(stats[x], stats[y]))
                        stats[x].append(ts_calc(stats[x]))
                        stats[x].append(oer_calc(stats[x]))

                        if len(stats[y]) < 28:
                            stats[y].append(possessions_calc(stats[y], stats[x]))
                            stats[y].append(ts_calc(stats[y]))
                            stats[y].append(oer_calc(stats[y]))

            # check to see if team has already been added to averages
            team_num = -1

            for z in range(0, len(team_avg)):
                if team_avg[z].count(stats[x][3]) != 0:
                    team_num = z
                    break
            if team_num == -1:
                team_avg.append([])
                team_avg[len(team_avg) - 1].append(stats[x][3])
                for z in range(8, len(stats[x])):
                    if stats[x][z] == "-":
                        team_avg[len(team_avg) - 1].append(0)
                    else:
                        team_avg[len(team_avg) - 1].append(float(stats[x][z]))
                team_avg[len(team_avg) - 1].append(1)

            else:
                for z in range(8, len(stats[x])-1):
                    if stats[x][z] == "-":
                        team_avg[team_num][z - 7] = 0
                    else:
                        try:
                            team_avg[team_num][z - 7] = float(team_avg[team_num][z - 7]) + float(stats[x][z])
                        except:
                            print("fook me laddy")
                team_avg[team_num][len(team_avg[team_num]) - 1] += 1
                games += 1

            if len(stats[x]) < 27:
                continue
        print(games)
        tempstring = stats_file[:len(stats_file)-4] + "_adv.csv"
        with open("C:/local/nba_stats/" + tempstring,'w',newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(stats)
            #for line in stats:
            #    csvwriter.writerow(line)
            #csvfile.writelines("\n")


        #calculate averages
        for x in range(0,len(team_avg)-1):
            for y in range(1,len(team_avg[x])-1):
                team_avg[x][y] = float(team_avg[x][y]) / team_avg[x][len(team_avg[x])-1]
            tempstring = stats_file[:len(stats_file) - 4] + "_avg.csv"
        with open("C:/local/nba_stats/" + tempstring, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(team_avg)


    else:
        continue