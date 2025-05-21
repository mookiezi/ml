input_file = "tagged.txt"
output_file = "oversampled.txt"

# Ask for the oversample amount
oversample_amount = int(input("Enter the oversample amount for <GOOD> lines: "))

with open(input_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

good_lines = [l for l in lines if l.startswith("<GOOD>")]
bad_lines = [l for l in lines if l.startswith("<BAD>")]

# Oversample good lines based on user input
oversampled_good = good_lines * oversample_amount

# Now combine the oversampled GOOD lines with BAD lines in the final output
final_lines = []
for i in range(min(len(oversampled_good)//oversample_amount, len(bad_lines))):
    final_lines.extend([good_lines[i]]*oversample_amount)  # Add oversampled <GOOD> lines
    final_lines.append(bad_lines[i])  # Add 1 <BAD> line

# If there are extra good lines left, append them to the final output
final_lines.extend(oversampled_good[len(bad_lines)*oversample_amount:])

# If there are extra bad lines left, append them to the final output
final_lines.extend(bad_lines[len(oversampled_good)//oversample_amount:])

with open(output_file, "w", encoding="utf-8") as f_out:
    for line in final_lines:
        f_out.write(line + "\n")

print(f"💾 Saved {len(final_lines)} lines to {output_file}")
