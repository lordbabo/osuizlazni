words_dictionary = {}
words=[]

song = open('OSU_LV1/song.txt')

for line in song:
    line = line.rstrip()
    words = words + line.split()

for word in words:
    if word in words_dictionary:
        words_dictionary[word]+=1
    else:
        words_dictionary[word]=1

uniq_words_count=0

for word in words_dictionary:
    if words_dictionary[word]==1:
        uniq_words_count+=1
        print(f"{word}:{words_dictionary[word]}")
      
print("Broj rijeci koje se pojavljuju samo jednom:",uniq_words_count)
