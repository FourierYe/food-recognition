from skimage import io
import numpy as np

img = io.imread('data/food-101/images/grilled_cheese_sandwich/36521.jpg')
io.imshow(img)
io.show()

img_red = img.copy()
img_green = img.copy()
img_blue = img.copy()

img_red[:,:,1:3] = 0
io.imshow(img_red)
io.show()

img_green[:,:,[0,2]] = 0
io.imshow(img_green)
io.show()

img_blue[:,:,[0,1]] = 0
io.imshow(img_blue)
io.show()