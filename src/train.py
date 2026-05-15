import os
import torch
import argparse
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizerFast, AutoTokenizer
from model import SILPlusPlusModel
from dataloader import SILPlusPlusDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="concode", choices=["concode", "defects4j"])
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 必须使用 Fast Tokenizer 以支持 return_offsets_mapping
    print("Loading Tokenizers...")
    t5_tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base", use_fast=True)
    bert_tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")

    print(f"Loading Dataset for {args.task}...")
    train_dataset = SILPlusPlusDataset(
        data_split="train", t5_tokenizer=t5_tokenizer, 
        bert_tokenizer=bert_tokenizer, task=args.task
    )
    dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    print("Initializing SIL++...")
    model = SILPlusPlusModel(
        "Salesforce/codet5-base", "microsoft/codebert-base", N=3, K=16
    ).to(device)

    # 冻结预训练骨干网络 (Staged Co-tuning)
    for name, param in model.generator.named_parameters(): param.requires_grad = False
    for name, param in model.discriminator.named_parameters(): param.requires_grad = False
    
    # 仅放开 SIL++ 融合层和解码器
    for param in model.fusion_blocks.parameters(): param.requires_grad = True
    for param in model.align_mlp.parameters(): param.requires_grad = True
    model.P.requires_grad = True
    for param in model.generator.decoder.parameters(): param.requires_grad = True

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5)
    scaler = torch.cuda.amp.GradScaler()
    lambda_align = 0.1

    print("🚀 Starting Training Loop...")
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                loss_gen, loss_align = model(**batch)
                loss = loss_gen + lambda_align * loss_align
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Align": f"{loss_align.item():.4f}"})

        # Save checkpoint
        os.makedirs("./checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"./checkpoints/silpp_{args.task}_epoch{epoch+1}.pt")

if __name__ == "__main__":
    main()