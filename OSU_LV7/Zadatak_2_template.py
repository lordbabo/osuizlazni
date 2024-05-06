import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
for i in range(1, 7): #postupak za sve slike
    img = Image.imread(f"OSU_LV7/imgs/test_{i}.jpg")

    # prikazi originalnu sliku
    plt.figure()
    plt.title("Originalna slika")
    plt.imshow(img)
    plt.tight_layout()
    plt.show()

    # pretvori vrijednosti elemenata slike u raspon 0 do 1
    img = img.astype(np.float64) / 255

    # transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))

    # rezultatna slika
    img_array_aprox = img_array.copy()

    #2.1 
    colors=np.unique(img_array,axis=0)
    print("Broj boja:",len(colors))

    j=[]
    #2.2
    for i in range(1,11):
        km=KMeans(n_clusters=i,init='k-means++',n_init=5,random_state=0) #2.4 mijenjanjem broja grupa k mijenja se broj grupa boja u slici, mijenjamo kvantizaciju
        km.fit(img_array_aprox)
        labels=km.predict(img_array_aprox)
        centroids = km.cluster_centers_
        j.append(km.inertia_)

    img_array_aprox=centroids[labels].reshape((w,h,d))

    plt.figure()
    plt.imshow(img_array_aprox)
    plt.show()
    plt.plot(range(1,11),j) #ovisnost j o k
    plt.show()

    #2.6 optimalan broj grupa je 3-4
        