import json
import ast
f = open("events.json")
lines = f.readlines()
list_json = []
for line in lines[0:10]:
    line2 = ast.literal_eval(line)
print(line2.keys())
print(line2['collection_slug'])