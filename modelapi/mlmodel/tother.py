import json

with open('new_states.json') as f:
  data = json.load(f)
print("this is the data",data,"len is",len(data))
