import time
import selenium
import os
from selenium.webdriver import Chrome

def scrape_nba(url):

    tempstring = url_entry.find("Season")
    tempstring = url_entry[tempstring:tempstring + 14]
    tempstring = tempstring.replace('=', '')
    if url.find("team") > -1:
        tempstring = "C:/local/nba_stats/teams_" + tempstring.replace('-','_') + ".csv"
    else:
        tempstring = "C:/local/nba_stats/" + tempstring.replace('-','_') + ".csv"
    if os.path.exists(tempstring):
        return
    driver = Chrome(executable_path='C:/local/chromedriver_win32/chromedriver.exe')
    driver.get(url)
    driver.find_element_by_class_name('run-it').click()
    done = False
    time.sleep(10)
    more_results = driver.find_element_by_class_name('table-addrows__button')

    while not done:
        try:
            for x in range(32000):
                more_results.click()
                time.sleep(0.1)
            break
        except selenium.common.exceptions.StaleElementReferenceException:
            done = True
    tabletest = driver.find_element_by_class_name('nba-stat-table').text
    linecount = 0
    with open(tempstring,'w') as csvfile:
        for line in tabletest.splitlines():
            tempstring = line.replace(' ', ',')
            tempstring = tempstring.replace('PLAYER', 'ID,FIRST,LAST')
            tempstring = tempstring.replace('MATCHUP', 'PLAYER TEAM,HOME,OPPONENT')

            if linecount != 0:
                tempstring = str(linecount) + "," + tempstring
            csvfile.writelines(tempstring + "\n")
            linecount+=1

    driver.close()


url_repo = list()

#for x in range(83,98):
for x in range(83,98):
    url = "https://stats.nba.com/search/player-game/#?sort=GAME_DATE&dir=-1&Season=19" + str(x) + "-" + str(x+1) + "&CF=PTS*gt*0"
    url_repo.append(url)
    url = "https://stats.nba.com/search/team-game/#?CF=PTS*gt*0&sort=GAME_DATE&dir=1&Season=19" + str(x) + "-" + str(x + 1)
    url_repo.append(url)

url_repo.append("https://stats.nba.com/search/player-game/#?sort=GAME_DATE&dir=-1&Season=1999-00&CF=PTS*gt*0")
url_repo.append("https://stats.nba.com/search/team-game/#?CF=PTS*gt*0&sort=GAME_DATE&dir=1&Season=1999-00")

for x in range(00,20):
    if x < 9:
        url = "https://stats.nba.com/search/player-game/#?sort=GAME_DATE&dir=-1&Season=200" + str(x) + "-0" + str(x+1) + "&CF=PTS*gt*0"
        url_repo.append(url)
        url = "https://stats.nba.com/search/team-game/#?CF=PTS*gt*0&sort=GAME_DATE&dir=1&Season=200" + str(x) + "-0" + str(x + 1)
    elif x == 9:
        url = "https://stats.nba.com/search/player-game/#?sort=GAME_DATE&dir=-1&Season=2009-10&CF=PTS*gt*0"
        url_repo.append(url)
        url = "https://stats.nba.com/search/team-game/#?CF=PTS*gt*0&sort=GAME_DATE&dir=1&Season=2009-10"
    elif x > 10:
        url = "https://stats.nba.com/search/player-game/#?sort=GAME_DATE&dir=-1&Season=20" + str(x) + "-" + str(x + 1) + "&CF=PTS*gt*0"
        url_repo.append(url)
        url = "https://stats.nba.com/search/team-game/#?CF=PTS*gt*0&sort=GAME_DATE&dir=1&Season=20" + str(x) + "-" + str(x + 1)
    url_repo.append(url)

for url_entry in url_repo:
    scrape_nba(url_entry)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
