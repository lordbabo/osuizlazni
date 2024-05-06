import numpy as np
import matplotlib . pyplot as plt

img = plt . imread ("OSU_LV2/road.jpg")

#a
plt.imshow(img, cmap='gray', alpha=0.7)
plt.title('Posvijetljena slika')
plt.show()

#b
width=len(img[0])
q_width=int(width/4)
plt.imshow(img[:, 1*q_width: 2*q_width,:],cmap='gray')
plt.show()

#c
rotated_img=np.rot90(img,3)
plt.imshow(rotated_img,cmap='gray')
plt.show()

#d
flipped_img=np.flip(img,0)
plt.imshow(flipped_img,cmap='gray')
plt.show()
