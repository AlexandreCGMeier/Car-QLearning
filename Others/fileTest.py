fileName = "Others/minDQ.txt"

data = ["Hello", "World", "This is a new line"]

with open(fileName, 'a') as file:
    for item in data:
        # Write each item on a new line
        file.write(item + "\n")
        print(item)
print("Data has been written to the file.")