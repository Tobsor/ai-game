import os
import csv
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

uri = os.getenv("NEO4J_URI")
auth = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

alliance_cypher = """
    MATCH (f1:Faction {id: $id1})
    MATCH (f2:Faction {id: $id2})
    CREATE (f1)-[:ALLIED_WITH]->(f2)
    RETURN f1
"""

friendly_cypher = """
    MATCH (f1:Faction {id: $id1})
    MATCH (f2:Faction {id: $id2})
    CREATE (f1)-[:FRIENDLY_WITH]->(f2)
    RETURN f1
"""

enemy_cypher = """
    MATCH (f1:Faction {id: $id1})
    MATCH (f2:Faction {id: $id2})
    CREATE (f1)-[:ENEMIES_WITH]->(f2)
    RETURN f1
"""

disrespect_cypher = """
    MATCH (f1:Faction {id: $id1})
    MATCH (f2:Faction {id: $id2})
    CREATE (f1)-[:DISRESPECTFUL_WITH]->(f2)
    RETURN f1
"""

def create_faction(driver, faction_data):
    return driver.execute_query(
        """
        CREATE (f:Faction {id: $id, form: $form, race: $race, political_power: $political_power, army_power: $army_power, cooperative_rating: $cooperative_rating })
        RETURN f
        """,
        id=faction_data.get("id"),
        form=faction_data.get("form"),
        race=faction_data.get("race"),
        political_power=faction_data.get("political_power"),
        army_power=faction_data.get("army_power"),
        cooperative_rating=faction_data.get("cooperative_rating")
    )

def create_relation(driver, factionId, relatedFactions, cypher):
    if len(relatedFactions) == 0 or relatedFactions[0] == "":
        return
    
    for faction in relatedFactions:
        driver.execute_query(cypher,
            id1=factionId,
            id2=faction,
            database="neo4j",
        )

with GraphDatabase.driver(uri, auth=auth) as driver:
    driver.verify_connectivity()
    print("Connection established.")

    driver.execute_query("""
        MATCH (n)
        DETACH DELETE n
    """)

    print("DB cleared")

    with open('./data.csv', mode ='r') as file:    
        csvFile = csv.DictReader(file, delimiter=';')
    
        # driver.execute_query("""
        #     CREATE INDEX faction_id_index FOR (f:Faction) ON (f.id)
        # """)

        print("loaded csv data")

        faction_relations = []

        for f_data in csvFile:
            create_faction(driver=driver, faction_data=f_data)
            faction_relations.append((f_data.get("id"), {
                "alliances": f_data.get("allied_with").split(","),
                "friendly": f_data.get("friendly_with").split(","),
                "enemy": f_data.get("enemy_with").split(","),
                "disrepectful": f_data.get("disrepectful_with").split(","),
            }))

        print("created nodes")
        
        for relation in faction_relations:
            create_relation(
                driver=driver,
                factionId=relation[0],
                relatedFactions=relation[1].get("alliances"),
                cypher=alliance_cypher
            )

            create_relation(
                driver=driver,
                factionId=relation[0],
                relatedFactions=relation[1].get("friendly"),
                cypher=friendly_cypher
            )

            create_relation(
                driver=driver,
                factionId=relation[0],
                relatedFactions=relation[1].get("enemy"),
                cypher=enemy_cypher
            )

            create_relation(
                driver=driver,
                factionId=relation[0],
                relatedFactions=relation[1].get("disrepectful"),
                cypher=disrespect_cypher
            )
            print("relations created for " + relation[0])
        
        print("relations created")
