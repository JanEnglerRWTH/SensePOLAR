
import bertFuncs as func

# Which word do you want to analyze?
# This current version only supports words and not phrases (e.g. "ice cream")
#word = "right"
word = input("Please enter the word you want to analyze: ")
word = str(word)

# In which context?
#context = "your argument is right"
#context = "he looked right and left"
context = input("Please enter the context (~example sentence) that includes the word: ")
context = str(context)
# How many top-dimensions should be printed?
top_d=12

top_d = input("Please enter how many top-dimensions should be printed: ")
top_d = int(top_d)
axis_list = func.analyzeWord(word, context,numberPolar=top_d)

# A NEGATIVE value signalizes a connection to the LEFT Antonym
# A POSITIVE value signalizes a connection to the RIGHT Antonym