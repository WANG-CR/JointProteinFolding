import os
import json


with open("filtered.json", "r") as f:
    a = json.load(f)
    
list_id = a["result_set"]
print(len(list_id))

with open("filtered.txt", "w") as f:
    for i in list_id:
        f.write(f"{i},")