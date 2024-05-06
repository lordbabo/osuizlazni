numbers=[]
i=0
print("Unesite brojeve dok ne upisete Done")
while(True):
    x=input()
    if(x=="Done"):
        break
    if(x.isnumeric()):
        numbers.append(float(x))
        i=i+1
    elif():
        print("Niste unjeli broj")

numbers.sort
avg=sum(numbers)/len(numbers)

print("Broj unesenih brojeva je:",i)
print("Sredina brojeva je",avg)
print("Minimum je:",min(numbers))
print("Maksimum je:",max(numbers))