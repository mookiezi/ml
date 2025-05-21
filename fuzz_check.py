from fuzzywuzzy import fuzz

# Prompt for user input
string1 = input("Enter the first string: ")
string2 = input("Enter the second string: ")

# Calculate fuzzy match score
score = fuzz.partial_ratio(string1, string2)

# Output the score
print(f"{score}")
