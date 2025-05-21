from transformers import AutoTokenizer

model_name = "EleutherAI/gpt-neo-125M"  # Replace with your model name
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Manually define SOS and EOS tokens
sos_token = "<GOOD>"
eos_token = tokenizer.eos_token

print(f"The manually defined SOS token for {model_name} is: {sos_token}")
print(f"The default EOS token for {model_name} is: {eos_token}")