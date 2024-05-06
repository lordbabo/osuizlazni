print("Unesite broj između 0.0 i 1.0")
try:
    a=float(input())
    if a<0.0 or a>1.0:
        print("Broj nije iz odgovarajućeg intervala")
    if a>=0.9 and a<=1:
        print("A")
    elif a<0.9 and a>=0.8:
        print("B")
    elif a<0.8 and a>=0.7:
        print("C")
    elif a<0.7 and a>=0.6:
        print("D")
    elif a<0.6 and a>=0:
        print("F")
except:
    print("Unesena stvar nije broj")

    
