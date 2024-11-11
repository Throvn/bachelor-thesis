import json
import math
import os
from typing import Any, Literal
allSpacesSorted = json.load(open("./sortedSpaces.json"))
allProjectMetadata = json.load(open("./allSantimentCoins.json"))

def findSpace(symbol: str, start = 0, end = len(allSpacesSorted) - 1) -> Any | Literal[False]:
    if (start > end):
        return False

    index = start + (end - start) // 2
    currSymbol = allSpacesSorted[index]['symbol'].lower().strip()
    if symbol == currSymbol:
        return allSpacesSorted[index]
    if symbol > currSymbol:
        return findSpace(symbol, index + 1, end)
    if symbol < currSymbol:
        return findSpace(symbol, start, index - 1)

daoSlugs = []
for prj in allProjectMetadata:
    # print(prj['ticker'].lower().strip())
    if findSpace(prj['ticker'].lower().strip(), 0, len(allSpacesSorted) - 1):
        print(prj['slug'])
        daoSlugs.append(prj['slug'])

print("Number of DAOs found in Santiment:", len(daoSlugs))

allSantimentProjects = json.load(open("./allSantimentProjects.json"))

daoProjects = []
for project in allSantimentProjects:
    if project['slug'] in daoSlugs:
        # TODO: GO back and merge prices1 and prices2 and so on together... project.allPrices 
        daoProjects.append(project)
        # TODO: Filter out all DAOs with less then 64 time series entries.

print("Length of total santiment DAO projects", len(daoProjects))

daoProjectsWithMinLength = []
for project in daoProjects:
    if project
json.dump(daoProjects, open("./allSantimentDaos.json", "w"))
print("Written", len(daoProjects), "DAO projects to file './allSantimentDaos.json'")