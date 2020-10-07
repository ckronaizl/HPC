import os
import matplotlib
import numpy
import matplotlib.pyplot as plt

#process stats based on team & season
for stats_file in os.listdir("C:/local/nba_stats/"):

    if stats_file.find("adv") > -1:
        with open("C:/local/nba_stats/" + stats_file,'r') as seasonstats:
            rawstats = seasonstats.readlines()

        team_stats = []
        games = 0
        for line in rawstats:
            line = line[:len(line)-1]
            team_stats.append(line.split(','))


        #begin calculations
        averages_win = []
        averages_loss = []
        wins = 0
        loss = 0

        #initialize offensive efficiency, possessions, true shooting %, points, +/- if available
        averages_win.append(0)
        averages_win.append(0)
        averages_win.append(0)
        averages_win.append(0)
        averages_win.append(0)

        averages_loss.append(0)
        averages_loss.append(0)
        averages_loss.append(0)
        averages_loss.append(0)
        averages_loss.append(0)

        for x in range (0, len(team_stats)-1):
            if team_stats[x][6] == "W":
                if len(team_stats[x]) > 27:
                    wins += 1
                    averages_win[0] += float(team_stats[x][29])
                    averages_win[1] += float(team_stats[x][27])
                    averages_win[2] += float(team_stats[x][28])
                    averages_win[3] += float(team_stats[x][8])
                if team_stats[x][26].find("-") == -1 and team_stats[x][26] != 0:
                    averages_win[4] += float(team_stats[x][26])

            elif team_stats[x][6] == "L":
                if len(team_stats[x]) > 27:
                    loss += 1
                    averages_loss[0] += float(team_stats[x][29])
                    averages_loss[1] += float(team_stats[x][27])
                    averages_loss[2] += float(team_stats[x][28])
                    averages_loss[3] += float(team_stats[x][8])
                if team_stats[x][26].find("-") == -1 and team_stats[x][26] != 0:
                    averages_loss[4] += float(team_stats[x][26])

        #get averages
        for x in range(0, len(averages_win)):
            averages_win[x] = averages_win[x] / wins

        for x in range(0, len(averages_loss)):
            averages_loss[x] = averages_loss[x] / loss

        # build plots
        teams_wins_labels = ["OER", "POSS", "TS%", "PTS", "+/-"]
        width = 0.35
        x = numpy.arange(len(teams_wins_labels))


        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, averages_win, width, label='Wins')
        rects2 = ax.bar(x + width/2, averages_loss, width, label='Losses')

        ax.set_ylabel('Qty')
        ax.set_title('W/L for ' + str(stats_file))
        ax.set_xticks(x)
        ax.set_xticklabels(teams_wins_labels)
        ax.legend()

        fig.tight_layout()

        tempstring = stats_file[:len(stats_file) - 4] + "_adv_comp.jpg"
        plt.savefig("C:/local/nba_stats/figs/" + tempstring)
        plt.close()
        #plt.show()



#process stats based on team & season averages
for stats_file in os.listdir("C:/local/nba_stats/"):

    if stats_file.find("adv") > -1:
        with open("C:/local/nba_stats/" + stats_file,'r') as seasonstats:
            rawstats = seasonstats.readlines()

        avg_stats = []
        games = 0
        for line in rawstats:
            line = line[:len(line)-1]
            avg_stats.append(line.split(','))


print("done")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
