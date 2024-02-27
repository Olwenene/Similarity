import spacy

# Load the language models
nlp_sm = spacy.load('en_core_web_sm')
nlp_md = spacy.load('en_core_web_md')

# Define the text
text = "A cat is chasing a monkey. The monkey is eating a banana."

# Process the text using both models
doc_sm = nlp_sm(text)
doc_md = nlp_md(text)

# Print the similarities
print("Similarities between 'cat', 'monkey', and 'banana':")
print("en_core_web_sm:")
for token1 in doc_sm:
    for token2 in doc_sm:
        if token1.text != token2.text:
            similarity = token1.similarity(token2)
            print(f"{token1.text} - {token2.text}: {similarity}")
            
print("\nen_core_web_md:")
for token1 in doc_md:
    for token2 in doc_md:
        if token1.text != token2.text:
            similarity = token1.similarity(token2)
            print(f"{token1.text} - {token2.text}: {similarity}")

# Write a note about the differences
note = """
The main difference between 'en_core_web_sm' and 'en_core_web_md' is the size of their
vocabulary and the dimensionality of their word vectors. 'en_core_web_md' includes
word vectors for more words and uses higher-dimensional vectors, which allows it to
capture more semantic nuances and similarities between words compared to 'en_core_web_sm'.
"""
print("\nNote:")
print(note)