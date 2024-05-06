import numpy as np
import matplotlib.pyplot as plt

black = np.zeros((50,50))
white= np.ones((50,50))

top=np.hstack((black,white))
bot=np.hstack((white,black))

box=np.vstack((top,bot))
plt.imshow(box,cmap='gray')
plt.show()