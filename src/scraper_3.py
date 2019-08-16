import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import time
import random
import csv
from time import sleep
from random import randint
from sys import argv

team_abbrs = ['ARI', 'ATL', 'BAL']
# , 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET', 'HOU', 'KCR', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SDP', 'SFG', 'SEA', 'STL', 'TBR', 'TEX', 'TOR', 'WSN', 'LAA']
years = ['2019', '2018']
# , '2017', '2016', '2015', '2014']
def scraper(team_abbrs, years, info=argv[1], outfile=argv[2]):
    '''This function scrapes team data from baseball-reference.com
    create a list of teams and dates to pass the function for the team_abbrs
    and years arguments.  info should either be 'b' or 'p' for batting or
    pitching stats. The outfile argument is simply what you want to name the file'''
    with open(outfile, 'w') as f:
        for z in team_abbrs:
            for y in years:
                w = csv.writer(f)

                url = 'https://www.baseball-reference.com/teams/tgl.cgi?team='+ z + '&t=' + info + '&year=' + y
                sleep(randint(3,5))
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                if info == 'b':
                    stats = soup.select('#team_batting_gamelogs tr')
                elif info == 'p':
                    stats = soup.select('#team_pitching_gamelogs tr')
                else:
                    print('Please enter only b or p as arguments')
                    break
                heads = list(stats[0].find_all('th'))
                headers = [head.text for head in heads[1:]]
                headers.extend(['year','team'])
                w.writerow(headers)
                for x in stats:
                    cell_var = []
                    cells = x.find_all('td')
                    if len(cells) > 0:
                        for idx, i in enumerate(headers[:-2]):
                            i = cells[idx].text
                            cell_var.append(i)
                        cell_var.append(y)
                        cell_var.append(z)
               
                    w.writerow(cell_var)

if __name__ == '__main__':
    scraper(team_abbrs, years)