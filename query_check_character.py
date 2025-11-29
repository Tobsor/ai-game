import csv
import keyboard
from classes.Character import Character

selected = 0
all_characters = []
situation="{{user}} enters the village of Rack and stumbles upon {{char}}. {{char}} initiates the contact to {{user}}"

def show_menu():
    global selected
    print("\n" * 30)
    print("Choose to talk to character:")
    for i, char in enumerate(all_characters):
        print("{0} {1} {2} {3}".format(">" if selected == i else " ", char.get("name"), char.get("faction"), "<" if selected == i else " "))

def up():
    global selected
    if selected == 0:
        selected = len(all_characters) - 1
    else:
        selected -= 1
    show_menu()

def down():
    global selected
    if selected == 4:
        selected = 0
    else:
        selected += 1
    show_menu()    

with open('./data/character_data_cop.csv', mode ='r') as file:
        csvFile = csv.DictReader(file, delimiter=';')

        print("retrieved factions")

        for character in csvFile:
            all_characters.append({ "name": character.get("name"), "faction": character.get("faction")})
show_menu()

loop = True

keyboard.add_hotkey('up', up)
keyboard.add_hotkey('down', down)

while loop:
    if keyboard.read_key() == "enter":
        loop = False

keyboard.remove_hotkey("up")
keyboard.remove_hotkey("down")

npc = Character(all_characters[selected], situation)

talk_ongoing = True
while talk_ongoing:
    answer = npc.prompt(prompt=input())
    print(answer)