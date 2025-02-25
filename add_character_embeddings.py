import csv
from classes.Character import Character

def add_character_embeddings():
    with open('./data/character_data.csv', mode ='r') as file:
        csvFile = csv.DictReader(file, delimiter=';')

        print("retrieved factions")

        for character in csvFile:
            faction = character.get("faction")
            name = character.get("name")

            new_char = Character(faction=faction, name=name)

            new_char.add_attribute("job", character.get("job"))
            new_char.add_attribute("past", character.get("past"))
            new_char.add_attribute("loyalty", character.get("loyalty"))
            new_char.add_attribute("helping", character.get("helping"))
            new_char.add_attribute("dislikes", character.get("dislikes"))
            new_char.add_attribute("likes", character.get("likes"))
            new_char.add_attribute("faction_loyalty", character.get("faction_loyalty"))
            new_char.add_attribute("knowlegdeable", character.get("knowlegdeable"))
            new_char.add_attribute("intelligence", character.get("intelligence"))

            print("added attributes for " + name)