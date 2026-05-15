import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, RobertaModel

class DeepGatedFusionBlock(nn.Module):
    """Deep Gated Fusion Layer (Eq 4-8)"""
    def __init__(self, d_t=768, heads=12):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_t, num_heads=heads, batch_first=True)
        self.W_g = nn.Linear(d_t * 2, d_t)
        self.W_u = nn.Linear(d_t * 2, d_t)
        self.layer_norm1 = nn.LayerNorm(d_t)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_t, d_t * 4),
            nn.GELU(),
            nn.Linear(d_t * 4, d_t)
        )
        self.layer_norm2 = nn.LayerNorm(d_t)

    def forward(self, H, A, mask=None):
        Z, _ = self.cross_attn(query=H, key=A, value=A, key_padding_mask=mask)
        HZ_concat = torch.cat([H, Z], dim=-1)  
        g = torch.sigmoid(self.W_g(HZ_concat))
        U = self.W_u(HZ_concat)                
        H_prime = self.layer_norm1(H + g * U)  
        return self.layer_norm2(H_prime + self.ffn(H_prime))

class SILPlusPlusModel(nn.Module):
    def __init__(self, generator_name, discriminator_name, N=3, K=16):
        super().__init__()
        self.generator = T5ForConditionalGeneration.from_pretrained(generator_name)
        self.discriminator = RobertaModel.from_pretrained(discriminator_name)
        
        d_t = self.generator.config.d_model
        d_b = self.discriminator.config.hidden_size
        
        self.align_mlp = nn.Sequential(nn.Linear(d_b, d_t), nn.GELU(), nn.Linear(d_t, d_t))
        self.fusion_blocks = nn.ModuleList([DeepGatedFusionBlock(d_t=d_t) for _ in range(N)])
        
        self.K = K
        if self.K > 0:
            self.P = nn.Parameter(torch.randn(1, self.K, d_t))
            self.bottleneck_attn = nn.MultiheadAttention(embed_dim=d_t, num_heads=12, batch_first=True)

    def forward(self, input_ids_t5, input_ids_bert, attention_mask_t5, span_alignment_matrix, labels=None):
        H_t5 = self.generator.encoder(input_ids=input_ids_t5, attention_mask=attention_mask_t5).last_hidden_state
        H_bert = self.discriminator(input_ids=input_ids_bert).last_hidden_state
        
        H_bert_aligned = torch.bmm(span_alignment_matrix, H_bert) 
        A = self.align_mlp(H_bert_aligned)
        
        H = H_t5
        for block in self.fusion_blocks:
            H = block(H, A)
            
        if self.K > 0:
            batch_size = H.size(0)
            H_bn, _ = self.bottleneck_attn(query=self.P.expand(batch_size, -1, -1), key=H, value=H)
            decoder_context = torch.cat([H_t5, H_bn], dim=1)
        else:
            decoder_context = H
            
        outputs = self.generator(encoder_outputs=(decoder_context,), labels=labels)
        loss_gen = outputs.loss
        
        # Alignment Loss calculation
        loss_align = (1.0 - (F.normalize(H_t5, p=2, dim=-1) * F.normalize(A, p=2, dim=-1)).sum(dim=-1)).mean()
        
        return loss_gen, loss_align