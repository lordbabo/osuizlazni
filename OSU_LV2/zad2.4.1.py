import numpy as np
import matplotlib . pyplot as plt
x = np.array([1,3,3,2,1])
y= np.array([1,1,2,2,1])
plt . plot (x , y , 'r', linewidth =2 , marker ="x", markersize =10 )
plt . axis ([0 ,4 ,0 , 4])
plt . xlabel ('x')
plt . ylabel ('y')
plt . title ('slika')
plt . show ()