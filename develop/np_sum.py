
def add_hist(h1, h2):
    if len(h1) < len(h2):
        h1 += [0] * (len(h2) - len(h1))
    for i in range(len(h1)):
        h1[i] += h2[i]

data = {}
for i in range(5):
    sum = []
    if (i != 3):
        add_hist(sum, [i + 1, i + 2])
    data[i] = sum

print(data)