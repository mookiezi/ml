# ────────────────────────────────────────────────────────────────────────────
#   LoRA-Enabled Causal Language Model Trainer for Hugging Face Transformers  
# ────────────────────────────────────────────────────────────────────────────

import os
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from transformers import get_linear_schedule_with_warmup
import random

# SEED -- Can comment out whole block to remove
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# SEED

def calculate_max_length(lines):
    return max(len(line.split()) for line in lines)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class TextLineDataset(Dataset):
    def __init__(self, lines, tokenizer, max_len):
        self.lines = lines
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        encoding = self.tokenizer.encode_plus(
            line,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

def init_scheduler(optimizer, train_loader_len, epochs, accum_steps, warmup_ratio=0.1):
    total_steps = (train_loader_len // accum_steps) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    return scheduler

def evaluate(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["input_ids"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

    avg_loss = val_loss / len(val_loader)
    return avg_loss

def save_model(model, tokenizer, optimizer, scheduler, save_path, epoch):
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
        torch.save(epoch, os.path.join(save_path, "epoch.pt"))
        
        print(f"Model saved to {save_path}")
    except Exception as e:
        print(f"Error saving model checkpoint: {e}")

def load_checkpoint(device, optimizer=None, scheduler=None, checkpoint_path=None):
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model.resize_token_embeddings(len(tokenizer))
    epoch = 0

    if checkpoint_path:
        if optimizer is not None:
            optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "optimizer.pt"), map_location=device))
        if scheduler is not None:
            scheduler.load_state_dict(torch.load(os.path.join(checkpoint_path, "scheduler.pt"), map_location=device))
        epoch = torch.load(os.path.join(checkpoint_path, "epoch.pt"), map_location=device)

    return model, tokenizer, optimizer, scheduler, epoch


def natural_sort(files):
    import re
    def alphanum_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
    
    return sorted(files, key=alphanum_key)

def parse_args():
    parser = argparse.ArgumentParser(description="train a causal language model with lora")
    parser.add_argument("-i", "--input_method", type=int, choices=[1, 2, 3, 4], required=False, help="select training option (1 = single file, 2 = single file resume, 3 = folder of .txt files, 4 = resume with folder of .txt files)")
    parser.add_argument("-m", "--model_name", type=str, default="EleutherAI/gpt-neo-125M", help="base model name or path")
    parser.add_argument("-o", "--output_dir", type=str, help="where to save model")
    parser.add_argument("-lr", "--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("-vb", "--val_batch_size", type=int, default=32, help="validation batch size")
    parser.add_argument("-ep", "--epochs", type=int, default=5, help="num epochs")
    parser.add_argument("-a", "--accum_steps", type=int, default=4, help="gradient accumulation steps")
    parser.add_argument("-ml", "--max_length", type=int, default=512, help="max input length")
    parser.add_argument("-es", "--eval_strategy", type=str, default="epoch", choices=["epoch", "steps", "no"], help="evaluation strategy")
    parser.add_argument("-c", "--resume_checkpoint", type=str, help="path to resume checkpoint from")
    parser.add_argument("-r", "--resume_txt_number", type=int, help="resume from the nth .txt file in multi-file scenario")
    parser.add_argument("-t", "--train_path", type=str, help="path to training data file or folder")
    args = parser.parse_args()

    if not args.input_method:
        print("Enter input method: 1 = Single file, 2 = Single file resume, 3 = Multi file, 4 = Multi file resume")
        args.input_method = int(input("Enter input method (1, 2, 3, or 4): ").strip())

    if not args.output_dir:
        args.output_dir = input("Enter model save path: ").strip()
    if not args.train_path:
        args.train_path = input("Enter path to training data folder: ").strip()
    if args.input_method == 2 and not args.resume_checkpoint:
        args.resume_checkpoint = input("Enter checkpoint path to resume from: ").strip()
    if args.input_method == 4:
        if not args.resume_checkpoint:
            args.resume_checkpoint = input("Enter checkpoint path to resume from: ").strip()
        if args.resume_txt_number is None:
            args.resume_txt_number = int(input("Enter resume txt number (the nth .txt file to resume from): ").strip())

    return args

def main():
    args = parse_args() # test

    print("Running with the following configuration:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not args.resume_checkpoint:
        model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        optimizer = None
        start_epoch = 0
        scheduler = None
    
        special_tokens_dict = {
                'bos_token': '<GOOD>',
                'additional_special_tokens': ['<GOOD>']
            }
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))

    if args.input_method == 1:
        print("Starting from scratch...")
        try:
            with open(args.train_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: Training file {args.train_path} not found.")
            return
        except IOError as e:
            print(f"Error reading file {args.train_path}: {e}")
            return

        max_len = args.max_length if args.max_length else calculate_max_length(lines)
        full_dataset = TextLineDataset(lines, tokenizer, max_len)
        val_size = max(int(0.1 * len(full_dataset)), 1)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size) 
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = init_scheduler(optimizer, len(train_loader), args.epochs, args.accum_steps)

        model.train()
        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1} learning rate: {get_lr(optimizer)}")
            epoch_loss = 0.0
            loop = tqdm(train_loader, desc=f"epoch {epoch+1}", dynamic_ncols=True)
            optimizer.zero_grad()

            for step, batch in enumerate(loop):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["input_ids"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()

                if (step + 1) % args.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                loop.set_postfix(loss=loss.item())
                epoch_loss += loss.item()

            print(f"epoch {epoch+1} avg loss: {epoch_loss / len(train_loader):.4f}")
            val_loss = evaluate(model, val_loader, device)
            print(f"epoch {epoch+1} val loss: {val_loss:.4f}")

        name = os.path.basename(args.train_path).replace(".txt", "")
        save_path = f"{args.output_dir}/{name}_epoch{epoch+1}"
        save_model(model, tokenizer, optimizer, scheduler, save_path, epoch + 1)
        print(f"\nCheckpoint saved as {args.output_dir}")

    elif args.input_method == 2:   
        try:
            with open(args.train_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: Training file {args.train_path} not found.")
            return
        except IOError as e:
            print(f"Error reading file {args.train_path}: {e}")
            return

        model, tokenizer, optimizer, scheduler, epoch = load_checkpoint(device, device, optimizer, scheduler, args.resume_checkpoint)

        max_len = args.max_length if args.max_length else calculate_max_length(lines)
        full_dataset = TextLineDataset(lines, tokenizer, max_len)
        val_size = max(int(0.1 * len(full_dataset)), 1)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size)
    
        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        if scheduler is None:
            scheduler = init_scheduler(optimizer, len(train_loader), args.epochs, args.accum_steps)
            
        print(f"Resuming from epoch {epoch + 1}")

        model.train()
        start_epoch = epoch
        for epoch in range(epoch, args.epochs):
            epoch_loss = 0.0
            loop = tqdm(train_loader, desc=f"epoch {epoch+1}", dynamic_ncols=True)
            optimizer.zero_grad()

            for step, batch in enumerate(loop):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["input_ids"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()

                if (step + 1) % args.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                loop.set_postfix(loss=loss.item())
                epoch_loss += loss.item()

            print(f"epoch {epoch+1} avg loss: {epoch_loss / len(train_loader):.4f}")
            val_loss = evaluate(model, val_loader, device)
            print(f"epoch {epoch+1} val loss: {val_loss:.4f}")

        name = os.path.basename(args.train_path).replace(".txt", "")
        save_path = f"{args.output_dir}/{name}_epoch{epoch+1}"
        save_model(model, tokenizer, optimizer, scheduler, save_path, epoch + 1)
        print(f"\nCheckpoint saved as {args.output_dir}")

    elif args.input_method == 3 or args.input_method == 4:
        files = [f for f in os.listdir(args.train_path) if f.endswith(".txt")]
        files = natural_sort(files)
        total_batches = 0
        
        for file in files:
            with open(os.path.join(args.train_path, file), "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            dataset_len = len(lines)
            batches = (dataset_len // args.batch_size) + (1 if dataset_len % args.batch_size != 0 else 0)
            total_batches += batches

        if args.input_method == 4:
            start_file_index = max(0, args.resume_txt_number - 1)
            model, tokenizer, optimizer, scheduler, epoch = load_checkpoint(device, optimizer, scheduler, args.resume_checkpoint)
            print(f"Resuming from epoch {start_epoch + 1}")
        else:
            start_file_index = 0
            start_epoch = 0

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
            total_steps = (total_batches // args.accum_steps) * args.epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps,
            )

        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        if scheduler is None:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps,
            )
        
        for file_idx, file in enumerate(files[start_file_index:], start=start_file_index):
            with open(os.path.join(args.train_path, file), "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            max_len = args.max_length if args.max_length else calculate_max_length(lines)
            full_dataset = TextLineDataset(lines, tokenizer, max_len)
            val_size = max(int(0.1 * len(full_dataset)), 1)
            train_size = len(full_dataset) - val_size
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
            model.train()
            epoch_start = start_epoch if file_idx == start_file_index else 0

            for epoch in range(epoch_start, args.epochs):
                optimizer.zero_grad()
                epoch_loss = 0.0
                loop = tqdm(train_loader, desc=f"epoch {epoch+1}", dynamic_ncols=True)
                for step, batch in enumerate(loop):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["input_ids"].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()

                    if (step + 1) % args.accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                    loop.set_postfix(loss=loss.item())
                    epoch_loss += loss.item()

                print(f"epoch {epoch+1} avg loss: {epoch_loss / len(train_loader):.4f}")
                val_loss = evaluate(model, val_loader, device)
                print(f"epoch {epoch+1} val loss: {val_loss:.4f}")

                name = os.path.basename(file).replace(".txt", "")
                save_path = f"{args.output_dir}/{name}_epoch{epoch+1}"
                save_model(model, tokenizer, optimizer, scheduler, save_path, epoch + 1)
                print(f"\nCheckpoint saved as {args.output_dir}")

            start_epoch = 0




if __name__ == "__main__":
    main()
