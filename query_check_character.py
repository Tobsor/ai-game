from data_ingestion import load_characters
import keyboard
from classes.Character import Character
from logger import configure_logging, get_logger

selected = 0
all_characters = []
situation="{{user}} enters the village of Rack and stumbles upon {{char}}. {{char}} initiates the contact to {{user}}"

configure_logging()
logger = get_logger(__name__)

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
    if selected >= len(all_characters) - 1:
        selected = 0
    else:
        selected += 1
    show_menu()    

all_characters = load_characters()
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

# npc.initiate_conversation()
    
