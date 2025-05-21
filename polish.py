import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import os
from tqdm import tqdm
import argparse
from safetensors.torch import save_model

def parse_args():
    """
    Parses command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="Train a causal language model with LoRA")
    parser.add_argument("-m", "--model_name", type=str, help="Base model name or path")
    parser.add_argument("-t", "--train_path", type=str, help="Path to training data")
    parser.add_argument("-e", "--eval_path", type=str, default=None, help="Optional eval data path")
    parser.add_argument("-o", "--output_dir", type=str, help="Where to save model")
    parser.add_argument("-lr", "--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("-ep", "--epochs", type=int, default=5, help="Num epochs")
    parser.add_argument("-a", "--accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("-c", "--resume_checkpoint", type=str, default=None, help="Path to resume checkpoint from")
    parser.add_argument("-ml", "--max_length", type=int, default=512, help="Max input length")
    parser.add_argument("-es", "--eval_strategy", type=str, default="epoch", choices=["epoch", "steps", "no"], help="Evaluation strategy")
    parser.add_argument("-r", "--resume_txt_number", type=int, default=None, help="Resume from the nth .txt file in multi-file scenario")
    parser.add_argument("-i", "--input_method", type=int, choices=[1, 2, 3, 4], help="Select input method (1=single file, 2=folder, 3=enter text, 4=.pt file)")

    args = parser.parse_args()

    # Input validation and prompting
    if not args.model_name:
        args.model_name = input("Enter model name: ").strip()
    if not args.output_dir:
        args.output_dir = input("Enter output directory path: ").strip()
    if not args.input_method:
        print("Enter input method:")
        print("1 = Single file")
        print("2 = Single file resume")
        print("3 = Multi file")
        print("4 = Multi file resume")
        args.input_method = int(input("Enter input method (1, 2, 3, or 4): ").strip())
    if args.input_method == 1 and not args.train_path:
        args.train_path = input("Enter path to training data file: ").strip()
    elif args.input_method == 2 and not args.train_path:
        args.train_path = input("Enter training data folder path: ").strip()

    return args


def prepare_dataset(lines, tokenizer, max_length):
    """
    Tokenizes and prepares the dataset for training.

    Args:
        lines (list): List of text lines.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        max_length (int): The maximum sequence length.

    Returns:
        list: A list of dictionaries, where each dictionary contains input_ids,
            attention_mask, and labels.
    """
    dataset = []
    for line in lines:
        tokenized = tokenizer(
            line,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        dataset.append({
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": tokenized["input_ids"].squeeze(0)
        })
    dataset.sort(key=lambda x: x["input_ids"].ne(tokenizer.pad_token_id).sum(), reverse=True)
    return dataset


def calculate_max_length(lines):
    """
    Calculates the maximum length of the input lines.

    Args:
        lines (list): List of text lines.

    Returns:
        int: The maximum length of the lines, or 64 if lines is empty.
    """
    return max(len(line.split()) for line in lines) if lines else 64


def freeze_model(model):
    """
    Freezes all parameters of the model.

    Args:
        model (torch.nn.Module): The model to freeze.
    """
    for param in model.parameters():
        param.requires_grad = False



def save_model_and_optimizer(model, tokenizer, optimizer, scheduler, output_dir, adapter=None):
    """
    Saves the model, tokenizer, optimizer, and scheduler.

    Args:
        model (torch.nn.Module): The model to save.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to save.
        output_dir (str): The directory where to save the files.
        adapter (torch.nn.Module, optional): The adapter model to save.
    """
    # Ensure output_dir points directly to the final location
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save the model using safetensors
    model_weights_path = os.path.join(output_dir, "model.safetensors")
    save_model(model, model_weights_path)

    # Save optimizer and scheduler states
    checkpoint_data = {
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint_data, os.path.join(output_dir, 'checkpoint.pt'))  # Save as .pt file

    # Save adapter model if present
    if adapter:
        adapter.save_pretrained(os.path.join(output_dir, 'adapter_model'))

    # Save config file.  This line is added for completeness.
    model.config.save_pretrained(output_dir)

    print(f"📝 Model, optimizer, scheduler, adapter, and config saved to {output_dir}")



def load_checkpoint(model, tokenizer, optimizer, scheduler, checkpoint_dir):
    """Loads the model, tokenizer, optimizer, and scheduler from a checkpoint.

    Args:
        model (torch.nn.Module): The model to load the state into.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to load.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to load the state into.
        checkpoint_dir (str): The directory where the checkpoint is located.

    Returns:
        tuple: A tuple containing the loaded model, tokenizer, optimizer, scheduler,
            and the starting epoch.
    """
    checkpoint_data = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.safetensors")))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
    start_epoch = checkpoint_data.get('epoch', 0)  # Default to 0 if not found
    return model, tokenizer, optimizer, scheduler, start_epoch



def load_adapter_model(model, adapter_model_path):
    """Loads an adapter model's state into the provided model.

    Args:
        model (torch.nn.Module): The main model to load the adapter into.
        adapter_model_path (str): Path to the adapter model's saved state.

    Returns:
        torch.nn.Module: The model with the adapter state loaded, or the original
            model if no adapter is found.
    """
    if os.path.exists(adapter_model_path):
        print(f"Loading adapter model from {adapter_model_path}...")
        #  It's more common to load the adapter weights *into* the main model
        #  rather than replacing the whole model state.  This assumes the adapter
        #  was saved separately.  If the entire model was saved with the adapter,
        #  then a regular load_state_dict is correct.  Without more context,
        #  it's not possible to be sure.  I'm assuming the common case of a
        #  separately saved adapter.  If this is wrong, the user can easily
        #  change it back.
        model.load_adapter(adapter_model_path) #  <---  ADDED THIS LINE
        # model.load_state_dict(torch.load(adapter_model_path)) #  <---  REMOVED THIS LINE
    else:
        print(f"No adapter model found at {adapter_model_path}, starting from scratch.")
    return model



def main():
    """Main function to train a causal language model with LoRA."""
    args = parse_args()

    # Print configuration
    print(f"Running with the following configuration:")
    print(f"Model: {args.model_name}")
    print(f"Train path: {args.train_path}")
    print(f"Eval path: {args.eval_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Accumulation steps: {args.accum_steps}")
    print(f"Resume checkpoint: {args.resume_checkpoint}")
    print(f"Max length: {args.max_length}")
    print(f"Eval strategy: {args.eval_strategy}")
    print(f"Resume from txt number: {args.resume_txt_number}")
    print(f"Input method: {args.input_method}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
    tokenizer.sos_token = "<GOOD>"   # Set custom SOS token
    tokenizer.add_special_tokens({
        'pad_token': tokenizer.eos_token,  # Add pad token
        'additional_special_tokens': ['<GOOD>']  # Add custom SOS token to additional special tokens
    })
    model.resize_token_embeddings(len(tokenizer))

    # Freeze the base model
    freeze_model(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # Common target modules for attention
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Initialize start_epoch
    start_epoch = 0

    # Load adapter model if it exists
    adapter_model_path = os.path.join(args.output_dir, "adapter_model") # Removed .safetensor
    model = load_adapter_model(model, adapter_model_path)

    # Print trainable parameters after loading adapter
    print("Trainable parameters after loading adapter:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    # Handle checkpoint loading
    if args.resume_checkpoint:
        checkpoint_dir = args.resume_checkpoint
        if os.path.exists(checkpoint_dir):
            print(f"Resuming from checkpoint: {checkpoint_dir}")
            model, tokenizer, optimizer, scheduler, start_epoch = load_checkpoint(
                model, tokenizer, optimizer, scheduler, checkpoint_dir)
            print(f"Resumed from epoch {start_epoch}")
        else:
            print("Checkpoint not found, starting from scratch.")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr) #Initialize here if not resuming
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95) #Initialize here if not resuming

    else:
        # Initialize optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    # Main training logic based on input method
    if args.input_method in [1, 2]:  # Single file or single file resume
        if args.input_method == 1:
            with open(args.train_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
        elif args.input_method == 2:
             with open(args.train_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

        max_len = calculate_max_length(lines)
        print(f"max_len: {max_len}")
        dataset = prepare_dataset(lines, tokenizer, max_length=max_len)

        if not dataset:
            print("No valid data.")
            return

        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        model.train()
        for epoch in range(start_epoch, args.epochs):
            epoch_loss = 0.0
            epoch_bar = tqdm(train_loader, desc=f"Polishing Epoch {epoch + 1}", dynamic_ncols=True, leave=True)
            for step, batch in enumerate(epoch_bar):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()

                if (step + 1) % args.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step() #update scheduler
                    optimizer.zero_grad()

                epoch_bar.set_postfix(loss=loss.item())
                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1} loss: {epoch_loss / len(train_loader):.4f}")

        # Save model, optimizer, scheduler, and adapter
        polished_model_dir = args.output_dir
        os.makedirs(polished_model_dir, exist_ok=True)
        save_model_and_optimizer(model, tokenizer, optimizer, scheduler, polished_model_dir, adapter=model)
        print(f"📝 Polished model and optimizer saved to {polished_model_dir}")

    elif args.input_method in [3, 4]:  # Multi-file training or resume
        folder_path = args.train_path #simplified
        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])

        if args.input_method == 4:
            if args.resume_txt_number is None:
                print("Error: --resume_txt_number is required with input_method 4.")
                return
            if args.resume_txt_number >= len(files):
                print("Error: resume_txt_number is too large.")
                return
            start_file_index = args.resume_txt_number
        else:
            start_file_index = 0

        for file_index in range(start_file_index, len(files)):
            file = files[file_index]
            print(f"Training on file: {file} (File {file_index + 1} of {len(files)})")
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            max_len = calculate_max_length(lines)
            print(f"max_len: {max_len}")
            dataset = prepare_dataset(lines, tokenizer, max_length=max_len)
            train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

            # Only load checkpoint at the beginning of the first file in a resumed session
            if args.input_method == 4 and file_index == start_file_index:
                model, tokenizer, optimizer, scheduler, start_epoch = load_checkpoint(
                    model, tokenizer, optimizer, scheduler, args.resume_checkpoint)
                print(f"Resuming from checkpoint {args.resume_checkpoint} at file {file_index+1}, epoch {start_epoch}")
            else:
                # Re-initialize optimizer and scheduler for each new file
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
                start_epoch = 0  # Reset epoch counter for new file


            model.train()
            for epoch in range(start_epoch, args.epochs): # Use start_epoch
                epoch_loss = 0.0
                epoch_bar = tqdm(train_loader, desc=f"Polishing Epoch {epoch + 1} ({file})", dynamic_ncols=True, leave=True)
                for step, batch in enumerate(epoch_bar):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    optimizer.zero_grad()
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()

                    if (step + 1) % args.accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                    epoch_bar.set_postfix(loss=loss.item())
                    epoch_loss += loss.item()
                print(f"Epoch {epoch + 1} loss: {epoch_loss / len(train_loader):.4f} (File: {file})")

            # Save after each file
            save_model_and_optimizer(model, tokenizer, optimizer, scheduler, args.output_dir, adapter=model)
            print(f"📝 Model and optimizer saved to {args.output_dir} after processing {file}")

    else:
        print("Error: Invalid input method.")
        return
    print("Training complete.")



if __name__ == "__main__":
    main()

