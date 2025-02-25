import keyboard
from classes.Character import Character

selected = 0

character_mapping = [
    {
        "name": "Tom",
        "faction": "ra",
        "context": "He loves to talk a lot. He is very focused on communicating about his issues and how he got poor. In that sense he loves to speak about himself rather than other issues. He knows only things about his near circumference but he will also try to make the impressionof . he is a simple man and knew only what he needed to know for fishingknowing anything, which might lead to him bluffing."
    },
    {
        "name": "Linda",
        "faction": "ra",
        "context": "She likes to talk with the common folks. She also is not shy to speak to foreign person of the same race. She is rather carefull talking with person of another race. She will now admit not knowing things if asked and will decline providing information about her privacy or family. she knows moderately much about what happens in the city and the people in their. She does not know a lot about outside her hometown and politics"
    },
    {
        "name": "Peter",
        "faction": "ra",
        "context": "He likes to talk. He loves to exaggarate and being overly positive about most things. He does not fear the consequences of his communication. He is very open about most topics. He reacts offended by critic. He is very knowledgeable about his people and the political state of his country. He is also very knowledgeable in food and drinks"
    },
    {
        "name": "Dirk",
        "faction": "ra",
        "context": "He fears the consequences his talks can have and is therefore not too talkative. He tries to provide information on a minimal basis and will keep infrormation for himself if he deems it critical information. He knows a lot about the world, ranging from his city, politics, society and so on. He is also very informed on what is happening outside of their kingdom, at least concerning infos that circulate. He undestands political relations from the factions very good"
    }
]

def show_menu():
    global selected
    print("\n" * 30)
    print("Choose to talk to character:")
    for i, char in enumerate(character_mapping):
        print("{0} {1} {2} {3}".format(">" if selected == i else " ", char.get("name"), char.get("faction"), "<" if selected == i else " "))

def up():
    global selected
    if selected == 0:
        selected = len(character_mapping) - 1
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

show_menu()

loop = True

keyboard.add_hotkey('up', up)
keyboard.add_hotkey('down', down)

while loop:
    if keyboard.read_key() == "enter":
        loop = False

keyboard.remove_hotkey("up")
keyboard.remove_hotkey("down")

npc = Character(
    character_mapping[selected].get("name"),
    character_mapping[selected].get("faction"),
    character_mapping[selected].get("context"),
)

while True:
    answer = npc.prompt(prompt=input())
    print(answer)

    prompt = answer
