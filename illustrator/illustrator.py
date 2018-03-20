def illustrate(text_file_path):
    pass

if __name__ == "__main__":
    # Imports
    import argparse
    import os

    # Get command line arguments
    parser = argparse.ArgumentParser(description='Produces illustrations for the given text.')
    parser.add_argument('--input-file', type=str, required=False,
                        default=os.path.join(os.path.dirname(__file__), '../data/peter_rabbit.txt'),
                        help='Path to the text file.')
    args = parser.parse_args()

    input_file = args.input_file

    illustrate(input_file)
