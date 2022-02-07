import json
import itertools

"""
params_info = {
    "x": [ 1, 2, 3],
    "y": [ "a", "b", "c"],
    "z": [ "A", "B", "C"],
    }

for param_vals in itertools.product(*params_info.values()):
    params = dict(zip(params_info.keys(), param_vals))
    data = {
      "A": params["x"],
      "B": "Green",
      "C": {
        "c_a": "O2",
        "c_b": params["y"],
        "c_c": ["D", "E", "F", params["z"]]
      }
    }
    jsonstr = json.dumps(data) # use json.dump if you want to dump to a file
    print(jsonstr)
    # add code here to do something with json
"""









