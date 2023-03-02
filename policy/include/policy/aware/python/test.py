import pickle

file_path = "/mnt/e/DatasetWareHouse/datasets/ptb/8415/input.pkl"


file_ = open(file_path, "rb")
data = pickle.load(file_)
print(len(data['G']))