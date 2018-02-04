# Install: pip install spacy && python -m spacy download en_vectors_web_lg
import os
import spacy

# Load English word vectors
nlp = spacy.load('en_vectors_web_lg')

def WordToVector(word):
    """
    Description
    -----------
        Gets the vector associated with a particular word (word2vec).

    Parameters
    ----------
        word : str
            The word to convert to vector form

    Returns
    -------
        word_vector : ndarray
            The vector associated with the input word
    """

    # Process the word
    doc = nlp(word)

    # Return the word vector
    return doc.vector

def WordListToVectors(filename):
    """
    Description
    -----------
        Gets the vectors associated with each word in a file (word2vec).
        Each line in the file should contain a single word or phrase.

    Parameters
    ----------
        filename : str
            The path to the file containing the words to convert to vector form
            Each line in the file should contain a single word or phrase

    Returns
    -------
        word_vectors : list
            A list of all the vectors associated with each word in the input file
    """

    # Store each word vector
    word_vectors = []

    # Read the input file
    with open(filename, 'r') as infile:

        # Each line of the input file should have a word or phrase to process
        for line in infile:

            # Process the word or phrase
            doc = nlp(line)

            # Save the word vector if the line is a single word or an average if it is a phrase.
            word_vectors.append(doc.vector)

    # Return a list of all the word vectors
    return word_vectors

if __name__ == '__main__':
    # Test for converting a single word to a vector
    word = 'airplane'
    word_vector = WordToVector(word)
    print(word_vector)

    # Test for converting a list of words in a file to a list of word vectors
    filename = os.path.join(os.path.dirname(__file__), '../data/quickdraw_categories.txt')
    word_vectors = WordListToVectors(filename)