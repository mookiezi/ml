import re

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        text = infile.read()

    # Regex to match blocks surrounded by matching quotes (supports multi-line)
    # This captures either double or single quoted blocks, including internal quotes
    pattern = r'(["\'])(.*?)\1'  # Non-greedy match between matching quotes

    matches = re.findall(pattern, text, flags=re.DOTALL)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for quote_char, block in matches:
            # block is the content inside outer quotes, preserve exactly as-is
            # Write <GOOD> + space + block + newline
            outfile.write(f"<GOOD> {block}\n")

if __name__ == "__main__":
    input_file = input("Please enter file name for input (without .txt): ")
    if not input_file.endswith(".txt"):
        input_file += ".txt"
    output_file_tagged = "all.txt"
    print(f"Processing '{input_file}' -> '{output_file_tagged}' ...")
    process_file(input_file, output_file_tagged)
    print("Done.")