ham_counter=0
spam_counter=0
ham_words=0
spam_words=0
spam_exclamation=0

file = open('OSU_LV1/SMSSpamCollection.txt')
for line in file:
    if line.startswith('ham'):
        ham_counter+=1
        ham_words+=len(line.split()[1:])
    elif line.startswith('spam'):
        spam_counter+=1
        spam_words+=len(line.split()[1:])
        if line.strip().endswith("!"):
            spam_exclamation+=1

average_ham=ham_words/ham_counter
average_spam=spam_words/spam_counter

print("Prosjecan broj rijeci u ham porukama:",average_ham)
print("Prosjecan broj rijeci u spam porukama:",average_spam)
print("Broj spam poruka koje zavrsavaju sa usklicnikom:",spam_exclamation)