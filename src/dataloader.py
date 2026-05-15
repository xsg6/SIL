import json
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from alignment import build_span_alignment_matrix

class SILPlusPlusDataset(Dataset):
    def __init__(self, data_split, t5_tokenizer, bert_tokenizer, task="concode", data_path=None):
        self.t5_tokenizer = t5_tokenizer
        self.bert_tokenizer = bert_tokenizer
        self.task = task
        
        # 1. 真实数据集加载逻辑
        if task == "concode":
            # 自动下载并加载 CodeXGLUE ConCode (Java)
            dataset = load_dataset("code_x_glue_ct_code_to_text", "java")
            self.data = dataset[data_split]
        elif task == "defects4j":
            # 加载预先提取好的 Defects4J JSONL 文件
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported task: {task}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 统一输入与输出格式
        if self.task == "concode":
            nl_text = item['docstring']
            target_code = item['code']
        else: # defects4j
            nl_text = item['buggy_code'] # APR 任务中输入为缺陷代码
            target_code = item['fixed_code']

        # Tokenize T5 (Generator)
        t5_inputs = self.t5_tokenizer(
            nl_text, max_length=256, padding='max_length', truncation=True, return_tensors="pt"
        )
        t5_labels = self.t5_tokenizer(
            target_code, max_length=256, padding='max_length', truncation=True, return_tensors="pt"
        )
        
        # Tokenize BERT (Discriminator)
        bert_inputs = self.bert_tokenizer(
            nl_text, max_length=256, padding='max_length', truncation=True, return_tensors="pt"
        )
        
        # 生成跨编码器对齐矩阵 (Span Alignment Matrix)
        span_matrix = build_span_alignment_matrix(
            nl_text, self.t5_tokenizer, self.bert_tokenizer, max_t5_len=256, max_bert_len=256
        )
        
        return {
            'input_ids_t5': t5_inputs['input_ids'].squeeze(),
            'attention_mask_t5': t5_inputs['attention_mask'].squeeze(),
            'input_ids_bert': bert_inputs['input_ids'].squeeze(),
            'span_matrix': span_matrix,
            'labels': t5_labels['input_ids'].squeeze()
        }