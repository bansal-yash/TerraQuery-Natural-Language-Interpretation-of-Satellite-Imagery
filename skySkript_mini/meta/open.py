import pickle

with open("/home/spandan/scratch/interiit/skySkript_mini/meta/a100011518_FI_17.pickle", "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data)
