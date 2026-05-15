import torch
import numpy as np

def build_span_alignment_matrix(text, t5_tokenizer, bert_tokenizer, max_t5_len=256, max_bert_len=256):
    """
    基于字符偏移量 (Character Offsets) 构建真实的分词器对齐矩阵。
    返回 shape: [max_t5_len, max_bert_len]
    """
    # 必须使用 Fast Tokenizer 以获取 offset_mapping
    t5_enc = t5_tokenizer(text, return_offsets_mapping=True, max_length=max_t5_len, truncation=True)
    bert_enc = bert_tokenizer(text, return_offsets_mapping=True, max_length=max_bert_len, truncation=True)
    
    t5_offsets = t5_enc['offset_mapping']
    bert_offsets = bert_enc['offset_mapping']
    
    matrix = np.zeros((max_t5_len, max_bert_len), dtype=np.float32)
    
    for i, t5_span in enumerate(t5_offsets):
        if t5_span == (0, 0): # 跳过特殊字符如 <s>, <pad>
            continue
            
        overlap_indices = []
        for j, bert_span in enumerate(bert_offsets):
            if bert_span == (0, 0): continue
            
            # 检测字符区间是否有交集: max(start1, start2) < min(end1, end2)
            if max(t5_span[0], bert_span[0]) < min(t5_span[1], bert_span[1]):
                overlap_indices.append(j)
                
        # Mean pooling 逻辑 (Eq 1): 若一个T5 token对应多个BERT token，权重均分
        if overlap_indices:
            weight = 1.0 / len(overlap_indices)
            for j in overlap_indices:
                matrix[i, j] = weight
                
    return torch.tensor(matrix)