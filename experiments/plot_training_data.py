import matplotlib.pyplot as plt
from matplotlib import patches

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

image = plt.imread("/kaggle/input/cardetection/car/train/images/000000_jpg.rf.b11f308f16626f9f795a148029c46d10.jpg")
plt.imshow(image)
img_h, img_w, _ = image.shape

with open('/kaggle/input/cardetection/car/train/labels/000000_jpg.rf.b11f308f16626f9f795a148029c46d10.txt', 'r') as f:
    cont = f.read()
    cls, x_c, y_c, h, w = map(float, cont.split(' '))
    cls = int(cls)

x_c *= img_w
y_c *= img_h
w *= img_w
h *= img_h

x1 = x_c - w / 2
y1 = y_c - h / 2
x2 = x_c + w / 2
y2 = y_c + h / 2

rect = patches.Rectangle((x1,y1), w, h, edgecolor = 'y', facecolor = 'none')
ax.add_patch(rect)
plt.show()

del image