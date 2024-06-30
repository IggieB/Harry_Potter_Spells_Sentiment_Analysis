A
import requests
from bs4 import BeautifulSoup

########### Global variables ###########
BASE_URL = "https://harrypotter.fandom.com/wiki/List_of_spells#"


def scrape_spells():
    """
    This function scrapes the overall list of spells in Harry Potter using
    the harry potter fandom wiki website.
    :return: the overall spells list
    """
    url = BASE_URL  # the website's URL
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # isolating the relevant elements in the html
    spell_elements = soup.select("h3 span i a")
    spell_names = [spell.text for spell in spell_elements]
    return spell_names

def save_spells(spells_list):
    """
    THis function saves the scraped list of spells as a txt file.
    :param spells_list: the scraped list
    :return: nothing, saves as a file
    """
    file = open('hp_spells_list.txt', 'w')
    for spell in spells_list:
        file.write(spell + "\n")
    file.close()

########### Call space ###########
spells_list = scrape_spells()
save_spells(spells_list)
