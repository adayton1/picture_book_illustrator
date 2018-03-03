# Install: pip install spacy && python -m spacy download en_core_web_lg
import os
import spacy

# Load English model
nlp = spacy.load('en_core_web_lg')


def get_noun_chunks(filename):

    # Process the whole text file
    text = open(filename).read()
    doc = nlp(text)

    # Save the nouns
    noun_chunks = []

    for chunk in doc.noun_chunks:
        # print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)
        noun_chunks.append(chunk)

    # Return a list of the noun chunks
    return noun_chunks


def get_nouns(filename):

    # Process the whole text file
    text = open(filename).read()
    doc = nlp(text)

    # Save the nouns
    nouns = []

    for chunk in doc.noun_chunks:
        # print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)
        token = chunk.root

        if token.lemma_ != "-PRON-":
            nouns.append((token.lemma_, token.vector, token.ent_type_))

    # Return a list of the nouns with their corresponding word vectors and the entity type (None if not an entity)
    return nouns


def get_entities(filename):
    # Process the whole text file
    text = open(filename).read()
    doc = nlp(text)

    # Save the entities
    entities = []

    # Print the entities
    for ent in doc.ents:
        #print(ent.text, ent.start_char, ent.end_char, ent.label_)
        entities.append((ent.text, ent.label_))

    return entities


if __name__ == '__main__':
    # Set the filename
    filename = os.path.join(os.path.dirname(__file__), '../data/peter_rabbit.txt')

    # Test for getting the noun chunks
    noun_chunks = get_noun_chunks(filename)

    for chunk in noun_chunks:
        print(chunk.text)
        print(chunk.root.text)
        print(chunk.root.ent_type_)
        print(chunk.root.head.text)
        print("\n")

    # Test for getting the nouns
    nouns = get_nouns(filename)

    for noun in nouns:
        print(noun[0], noun[2])

    # Test for getting the named entities
    entities = get_entities(filename)

    for entity in entities:
        print(entity[0], entity[1])
