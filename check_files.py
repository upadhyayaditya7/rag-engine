import os

data_dir = os.path.join(os.getcwd(), "data")
for file in os.listdir(data_dir):
    print(f"File found: '{file}'")
    