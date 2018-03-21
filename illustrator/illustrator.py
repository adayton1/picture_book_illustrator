from __future__ import unicode_literals
import codecs
import os
import subprocess
import spacy

# Load English model
nlp = spacy.load('en_core_web_lg')


def read_file(input_file):

    with codecs.open(input_file, "r", "utf-8") as f:
        text = f.read()

    pages = text.split("\n\n")

    return text, pages


def illustrate(input_file, output_dir):

    text, pages = read_file(input_file)

    # Process the whole doc
    # doc = nlp(text)

    # Iterate through each page
    for i, page in enumerate(pages):
        page_doc = nlp(page)

        nouns = []

        for chunk in page_doc.noun_chunks:
            noun = chunk.root

            if noun.lemma_ != "-PRON-":
                nouns.append(noun.lemma_)

        # Download image
        keywords = "{0}".format(" ".join(nouns))
        subprocess.call(["googleimagesdownload", "-k", keywords, "-l", "1", "-o", output_dir, "-f", "jpg"])

        # Move image to destination
        temp_dir = os.path.join(output_dir, keywords)
        temp = subprocess.check_output(["ls", temp_dir])[:-1]   # Throw away the \n character returned by ls
        temp = temp.decode("utf-8")
        filename, file_extension = os.path.splitext(temp)
        full_file_path = os.path.join(temp_dir, temp)
        destination = os.path.join(output_dir, "{0}".format(i) + file_extension)
        subprocess.call(["mv", full_file_path, destination])

        # Clear out the temporary folder
        subprocess.call(["rm", "-r", temp_dir])


if __name__ == "__main__":
    # Imports
    import argparse

    # Get command line arguments
    parser = argparse.ArgumentParser(description='Produces illustrations for the given text.')
    parser.add_argument('--input-file', type=str, required=False, help='Path to the text file.',
                        default=os.path.join(os.path.dirname(__file__), '../data/peter_rabbit.txt'))
    parser.add_argument('--output-dir', type=str, required=False, help='Path to the output directory.',
                        default=os.path.join(os.path.dirname(__file__), '../illustrated_books'))
    args = parser.parse_args()

    input_file = os.path.abspath(args.input_file)
    output_dir = os.path.abspath(args.output_dir)

    illustrate(input_file, output_dir)
