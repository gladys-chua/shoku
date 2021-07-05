import pandas as pd
from ast import literal_eval
import time
from collections import OrderedDict
import spacy

# Initialize spacy 'en_core_web_sm' model, keeping only tagger component needed for lemmatization
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Preprocessing
def remove_space_around_punctuation(ingredient_lemma):
    ingredient_lemma = ingredient_lemma.replace(" - ", "-")
    ingredient_lemma = ingredient_lemma.replace(" %", "%")
    return ingredient_lemma

# Tokenize
def tokenize(ingredients):
    word_list = []
    ingredients_lemma = []
    for recipe in ingredients:
        recipe_lemma = []
        for ingredient in recipe:
            ingredient = ingredient.replace("&", "and")
            doc = nlp(ingredient)
            ingredient_lemma = ""
            for token in doc:
                ingredient_lemma += token.lemma_ + " "
                ingredient_lemma = remove_space_around_punctuation(ingredient_lemma)
            word_list.append(ingredient_lemma.rstrip())
            recipe_lemma.append(ingredient_lemma.rstrip())
        ingredients_lemma.append(recipe_lemma)
    return word_list, ingredients_lemma

# inverted index dictionary
def inverted_index(word_list, ingredients_lemma, id):
    inv = {key:[] for key in word_list}
    for recipe in range(len(ingredients_lemma)):
        for ingredient in ingredients_lemma[recipe]:
            inv[ingredient].append(id[recipe])
    return inv

# Sort inverted index dictionary
def sorted_inverted_index(inv):
    inv_sorted = OrderedDict(sorted(inv.items()))
    return inv_sorted

# Read CSV file
start_time = time.time()
df = pd.read_csv("Project/archive/RAW_recipes.csv",nrows =50, converters={'ingredients':literal_eval})
ingredients = df['ingredients']
id = df['id']

# Get inverted index for df
word_list, ingredients_lemma = tokenize(ingredients)
inv = inverted_index(word_list, ingredients_lemma, id)
sorted_inv = sorted_inverted_index(inv)
print(sorted_inv)

# Find time taken
end_time = time.time()
time_taken = end_time - start_time
print(time_taken)
# >> 321.21118783950806 (for nrows = 100, not sorted)
# >> 285.59813237190247 ~ 335.7668788433075 (for nrows = 10000, sorted)
# >> 4457.500209569931 (for all rows )

#display words in ascending order
for word in sorted_inv.keys():
    print(f"{word} -> ", sorted_inv[word])