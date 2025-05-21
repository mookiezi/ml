import re
import nltk
from nltk.corpus import words

nltk.download('words')
english_words = set(words.words())

# Custom words to filter out (add more as needed)
filter_words = {"candy", "lady", "pony", "sup", "nsfw", "nov", "im", "real", "girl", "fake", "zombie", "big", "more", "king"}

# **Verification Function**
def verify_words(word_list):
    for word in word_list:
        if word.lower() not in english_words:
            print(f"'{word}' NOT found in nltk.corpus.words")
        else:
            print(f"'{word}' FOUND in nltk.corpus.words")

verify_words(["lady", "real", "fake", "zombie", "big", "more", "king"])  # Check the words in question

def clean_and_format_names(input_filename, output_filename):
    """
    Reads names from a file, attempts to convert "fancy" unicode characters
    to their closest ASCII equivalents, removes other symbols and numbers,
    filters out English words and unwanted sequences, and removes single
    letter words.
    Splits names into individual words,
    encloses each word in quotes, joins them with ", ", and saves
    the result to a new file.  It also adds line breaks after every
    comma if the line exceeds 65 characters.

    Args:
        input_filename (str): The name of the input file (txt or csv).
        output_filename (str): The name of the output file.
    """

    def insert_linebreaks(text, max_line_length=65):
        words_in_line = text.split(',')
        current_line = ''
        result = []

        for word in words_in_line:
            word = word.strip()  # Remove leading/trailing spaces from each word
            if len(current_line + word) + (1 if current_line else 0) <= max_line_length:
                current_line += (',' if current_line else '') + word
            else:
                result.append(current_line + ',')
                current_line = word
        result.append(current_line + ',')
        return '\n'.join(result).replace('\n,', '\n')

    def replace_and_remove_chars(text):
        # Extended character mapping
        char_map = {
            '𝐿': 'L', '𝒴': 'Y', '𝒩': 'N',
            '𖦹': '',  # Remove this symbol
            '𝔀': 'w', '𝓸': 'o', '𝓵': 'l', '𝓯': 'f', '𝓰': 'g', '𝓲': 'i', '𝓻': 'r',
            '𝖜': 'w',
            'ʏ': 'Y', '𝙤': 'o', '𝙝': 'h', '𝘢': 'a', '𝗻': 'n',
            '𝖅': 'Z', '𝖆': 'a', '𝖈': 'c', '𝖍': 'h',
            '𝕵': 'J', '𝖔': 'o', '𝖍': 'h', '𝖆': 'a', '𝖓': 'n', '𝙼': 'M', '𝚘': 'o', '𝚗': 'n', '𝚜': 's', '𝚝': 't', '𝚎': 'e', '𝚛': 'r',
            '𝓒': 'C', '𝒽': 'h', '𝒶': 'a', '𝓃': 'n', '𝑒': 'e',
            '𝕎': 'W',
            '𝔸': 'A', '𝔹': 'B', '𝗖': 'C', '𝔻': 'D', '𝔼': 'E', '𝔽': 'F', '𝔾': 'G', 'ℍ': 'H', '𝕀': 'I', '𝕁': 'J', '𝕂': 'K', '𝕃': 'L', '𝕄': 'M', 'ℕ': 'N', '𝕆': 'O', 'ℙ': 'P', 'ℚ': 'Q', 'ℝ': 'R', '𝕊': 'S', '𝕋': 'T', '𝕌': 'U', '𝕍': 'V', '𝕎': 'W', '𝕏': 'X', '𝕐': 'Y', 'ℤ': 'Z',
            '𝕒': 'a', '𝕓': 'b', '𝕔': 'c', '𝕕': 'd', '𝕖': 'e', '𝕗': 'f', '𝕘': 'g', '𝕙': 'h', '𝕚': 'i', '𝕛': 'j', '𝕜': 'k', '𝕝': 'l', '𝕞': 'm', '𝕟': 'n', '𝕠': 'o', '𝕡': 'p', '𝕢': 'q', '𝕣': 'r', '𝕤': 's', '𝕥': 't', '𝕦': 'u', '𝕧': 'v', '𝕨': 'w', '𝕩': 'x', '𝕪': 'y', '𝕫': 'z',
            'à': 'a', 'á': 'a', 'â': 'a', 'ä': 'a',
            'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
            'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
            'ò': 'o', 'ó': 'o', 'ô': 'o', 'ö': 'o',
            'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u',
            'ç': 'c', 'ñ': 'n',
            'ř': 'r', 'ą': 'a',
            'ō': 'o', 'ū': 'u'
        }
        text = "".join(char_map.get(char, char) for char in text)

        # Specific removals using regex
        text = re.sub(r'𓆏', '', text)  # Remove frogs
        text = re.sub(r'ᓚᘏᗢ', '', text)  # Remove cat face
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove all non-ASCII
        text = re.sub(r'[\u4e00-\u9fff]+', '', text) # Remove CJK characters
        text = re.sub(r'[\u0600-\u06FF]+', '', text) # Remove Arabic characters
        text = re.sub(r'ᚺᛖᚨᚢᛖᚾᛚᚤᛗᛟᛖ', '', text) # Remove specific runes
        return text

    try:
        with open(input_filename, 'r', encoding='utf-8') as infile:
            all_names = infile.read().splitlines()
    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.")
        return

    quoted_words = set()  # Use a set to track unique words
    for name in all_names:
        # Replace fancy characters and remove specific sequences
        name = replace_and_remove_chars(name)

        # Remove symbols and numbers
        cleaned_name = re.sub(r'[^\w\s]', '', name)
        cleaned_name = re.sub(r'\d+', '', cleaned_name)
        cleaned_name = cleaned_name.lower()

        # Remove leading spaces and spaces between single letters
        cleaned_name = cleaned_name.lstrip()
        cleaned_name = re.sub(r'(?<!\w)(\w)\s+(\w)(?!\w)', r'\1\2', cleaned_name)

        should_keep = True

        if cleaned_name in filter_words:
            should_keep = False
        elif cleaned_name in english_words:
            should_keep = False
        elif len(cleaned_name) <= 1:
            should_keep = False

        if should_keep:
            cleaned_words = cleaned_name.split()
            cleaned_name = "".join(cleaned_words)
            quoted_words.add(f'"{cleaned_name}"')

    output_string = ", ".join(quoted_words)
    formatted_output = insert_linebreaks(output_string)

    if not output_filename.endswith(".txt"):
        output_filename += ".txt"

    try:
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            outfile.write(formatted_output)
        print(f"Processed data saved to '{output_filename}'")
    except Exception as e:
        print(f"Error writing to output file: {e}")

if __name__ == "__main__":
    input_file = input("Enter the input filename: ")
    output_file = input("Enter the output filename: ")
    clean_and_format_names(input_file, output_file)
