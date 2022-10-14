import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random

scale = 10
size = 37
im = Image.open('square.jpg')

# Create figure and axes
fig, ax = plt.subplots()

# Display the image
ax.imshow(im)

# Create a Rectangle patch
for i in range(size+1):
    for j in range(size+1):
        print((0.5+i)/(size+1)*300, (0.5+j)/(size+1)*300)
        color = 'r' if (i+j)%4==0 else ('b' if (i+j)%4==1 else ('black' if (i+j)%4==2 else 'g'))
        rect = patches.Rectangle(((0.5+i)/(size+1)*300 + random.random()*2, (0.5+j)/(size+1)*300+random.random()*2), scale, scale, linewidth=0.5, edgecolor=color, facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)


plt.savefig('boxes_10_37.png')
plt.show()
