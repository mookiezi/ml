import os

def split_file(filename, chunk_size=5000, output_folder=None):
    with open(filename, "r", encoding="utf-8") as f:
        # Read whole file and split into blocks starting with <GOOD>
        content = f.read()
    
    # Split on '<GOOD>' but keep the delimiter by using a regex split with capture
    import re
    parts = re.split(r'(?=<GOOD>)', content)

    # Remove empty parts (could be before first <GOOD>)
    parts = [p.strip() for p in parts if p.strip()]

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    else:
        output_folder = os.getcwd()

    chunk_idx = 1
    chunk = []
    count = 0

    for part in parts:
        # Each 'part' is one full <GOOD> block
        chunk.append(part)
        count += 1

        if count >= chunk_size:
            chunk_filename = os.path.join(output_folder, f"{chunk_idx}.txt")
            with open(chunk_filename, "w", encoding="utf-8") as chunk_file:
                chunk_file.write("\n".join(chunk))  # keep blank line between blocks
            print(f"Created: {chunk_filename}")
            chunk_idx += 1
            chunk = []
            count = 0

    # Write remaining chunk if any
    if chunk:
        chunk_filename = os.path.join(output_folder, f"{chunk_idx}.txt")
        with open(chunk_filename, "w", encoding="utf-8") as chunk_file:
            chunk_file.write("\n".join(chunk))
        print(f"Created: {chunk_filename}")

if __name__ == "__main__":
    filename = input("Enter the filename to split: ")
    if not filename.endswith(".txt"):
        filename += ".txt"
    output_folder = input("Enter the output folder name (leave blank for current directory): ")
    split_file(filename, output_folder=output_folder)
