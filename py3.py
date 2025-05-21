import safetensors

# Try to load the file and inspect its metadata
file_path = "D:/ml/maniac/model.safetensors"
data = safetensors.load(file_path)
print(data.metadata)
