x = 23
print ( x )
x = x + 7
print ( x ) # komentar : ispis varijable na ekranu

x = 23
y = x > 10
print ( y )

x = 23
if x < 10:
    print (" x je manji od 10 ")
else :
    print (" x je veci ili jednak od 10 ")

i = 5
while i > 0:
    print ( i )
    i = i - 1
    print (" Petlja gotova ")

for i in range (0 , 5 ):
    print ( i )

lstEmpty = [ ]
lstFriend = ['Marko', 'Luka', 'Pero']
lstFriend . append ('Ivan')
print ( lstFriend [0])
print ( lstFriend [0:1:2])
print ( lstFriend [ :2])
print ( lstFriend [1: ])
print ( lstFriend [1:3])

a = [1 , 2 , 3]
b = [4 , 5 , 6]
c = a + b
print ( c )
print ( max ( c) )
c[0] = 7
c . pop ()
for number in c:
    print ('List number', number )
print ('Done !')
print(c[0:-1])

line = 'Dobrodosli u nas grad'
if( line . startswith ( 'Dobrodosli') ):
    print ('Prva rijec je Dobrodosli')
elif ( line . startswith ( 'dobrodosli') ):
    print ('Prva rijec je dobrodosli')
line . lower ()
print ( line )
data = 'From : pero@yahoo . com'
atpos = data . find ('@')
print ( atpos )

letters = ('a', 'b', 'c', 'd', 'e')
numbers = (1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 )
mixed = (1 , ' Hello ', 3.14 )
print ( letters [0])
print ( letters [1:4])
for letter in letters :
    print ( letter )

hr_num = {'jedan':1 , 'dva':2 , 'tri':3}
print ( hr_num )
print ( hr_num ['dva'])
hr_num ['cetiri'] = 4
print ( hr_num )