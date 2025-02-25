import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from classes.Faction import Faction

load_dotenv()

uri = os.getenv("NEO4J_URI")
auth = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

def add_faction_embeddings():
    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()
        print("Connection established.")

        all_factions = driver.execute_query("""
            MATCH (n) RETURN n.id as id, n.name as name, n.army_power as army_power, n.cooperative_rating as cooperative_rating, n.biological_properties as biological_properties, n.nature as nature, n.goal as goal,n.habital_area as habital_area, n.quantity as quantity, n.strengths as strengths, n.weakness as weakness
        """)

        print("retrieved factions")

        for d in all_factions.records:
            faction_data = d.data()

            new_faction = Faction(faction_data.get("id"), faction_data.get("name"))
            
            new_faction.add_attribute("biological_properties", faction_data.get("biological_properties"))
            new_faction.add_attribute("nature", faction_data.get("nature"))
            new_faction.add_attribute("strengths", faction_data.get("strengths"))
            new_faction.add_attribute("weakness", faction_data.get("weakness"))
            new_faction.add_attribute("goal", faction_data.get("goal"))
            new_faction.add_attribute("habital_area", faction_data.get("habital_area"))
            new_faction.add_attribute("quantity", faction_data.get("quantity"))

            print("added embeddings for " + faction_data.get("name"))