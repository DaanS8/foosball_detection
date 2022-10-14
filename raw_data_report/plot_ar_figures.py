import json
import numpy as np
import matplotlib.pyplot as plt

# JSON file
f = open('train/_annotations.coco.json', "r")
data = json.loads(f.read())

# Iterating through the json
# list
ars = []
for i in data['annotations']:
    if i['category_id'] == 2:
        width, height = i['bbox'][2:]
        ars.append(max(width/height, height/width))

ars = list(sorted(ars))[:-3]  # remove odd ones out, not representative and messes up plot
print('Median ar:', ars[int(len(ars)/2)])
# Closing file
f.close()

np_ars = np.array(ars)

fig = plt.figure(figsize=(16, 5))
ax = fig.add_subplot(111)
bp = ax.boxplot(np_ars, vert=0)
ax.set_xlabel('Aspect Ratio Figures')
plt.title("Boxplot aspect ratio of figures in the training set")
ax.set_yticks([])
# show plot
plt.savefig("ar_figures.png")
plt.show()

