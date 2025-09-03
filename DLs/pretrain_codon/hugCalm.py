
import os,sys
import torch
import json
import pickle 
import math
import torch.nn as nn
from torch import Tensor, nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer
from calm.multihead_attention import MultiheadAttention
from calm.model import ProteinBertModel
from calm.alphabet import Alphabet


def gelu(x):
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    class ESM1bLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    from torch.nn import LayerNorm as ESM1bLayerNorm

class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, weight):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = ESM1bLayerNorm(embed_dim)
        self.weight = nn.Parameter(weight.clone()) # weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x

class TransformerLayer(nn.Module):
    """Transformer layer block."""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        attention_dropout=0.,
        add_bias_kv=True,
        use_esm1b_layer_norm=False,
        rope_embedding=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout
        self.rope_embedding = rope_embedding
        self._init_submodules(add_bias_kv, use_esm1b_layer_norm)

    def _init_submodules(self, add_bias_kv, use_esm1b_layer_norm):
        BertLayerNorm = ESM1bLayerNorm if use_esm1b_layer_norm else nn.LayerNorm

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            dropout=self.attention_dropout,
            rope_embedding=self.rope_embedding,
        )
        self.self_attn_layer_norm = BertLayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = BertLayerNorm(self.embed_dim)

    def forward(
        self, x, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        if torch.isnan(x).any():
            print("jkfskdjksd")

        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        if torch.isnan(x).any():
            print("jkfskdjksd")

        return x, attn

class CaLMConfig(PretrainedConfig):
    model_type = "calm"  # 添加这行，确保一致

    def __init__(
        self,
        max_positions=1024,
        num_layers=12,
        embed_dim=768,
        attention_heads=12,
        ffn_embed_dim=3072,
        attention_dropout=0.0,
        logit_bias=False,
        rope_embedding=False,
        vocab_size=None,  # 由字母表大小决定
        pad_token_id=None,
        mask_token_id=None,
        cls_token_id=None,
        num_labels=2,  # 新增：默认值为 2（二分类任务）
        problem_type= None,
        eos_token_id=None,
        dropout_rate=0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_positions = max_positions
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_dropout = attention_dropout
        self.logit_bias = logit_bias
        self.rope_embedding = rope_embedding
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.cls_token_id = cls_token_id
        self.eos_token_id = eos_token_id
        self.num_labels = num_labels  # 新增
        self.problem_type = problem_type
        self.dropout_rate = dropout_rate


class CaLMModel(PreTrainedModel):
    
    config_class = CaLMConfig  # 必须加这一行，告诉 AutoModel 用哪个 config 类

    def __init__(self, config):
        super().__init__(config)
        # 创建词嵌入
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id
        )
        
        # Transformer层
        # from .modules import TransformerLayer
        self.layers = nn.ModuleList([
            TransformerLayer(
                config.embed_dim,
                config.ffn_embed_dim,
                config.attention_heads,
                attention_dropout=config.attention_dropout,
                add_bias_kv=False,
                use_esm1b_layer_norm=True,
                rope_embedding=config.rope_embedding
            )
            for _ in range(config.num_layers)
        ])

        self.embed_scale = 1
        self.emb_layer_norm_before = None
        self.emb_layer_norm_after = ESM1bLayerNorm(config.embed_dim)
        
        # 语言模型头
        self.lm_head = RobertaLMHead(
            embed_dim=config.embed_dim,
            output_dim=config.vocab_size,
            weight=self.embed_tokens.weight,
        )


    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        token_type_ids=None, 
        position_ids=None,
        repr_layers=[12],
        need_head_weights=False,
        return_dict=None
    ):
        if repr_layers is None:
            repr_layers = []
        
        # 处理输入
        if attention_mask is None:
            attention_mask = input_ids.eq(self.config.pad_token_id)   #ne
        else:
            attention_mask = (attention_mask == 0)
        
        # 嵌入层
        x = self.embed_scale * self.embed_tokens(input_ids)
        
        # 位置编码
        if not self.config.rope_embedding:
            x = x + self.embed_positions(input_ids)
        
        # 处理padding
        if attention_mask is not None:
            x = x * (1 - attention_mask.unsqueeze(-1).type_as(x))
        
        # 跟踪表示
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x
        
        # 跟踪注意力权重
        if need_head_weights:
            attn_weights = []
        
        # 序列维度转换 (B, T, E) -> (T, B, E)
        x = x.transpose(0, 1)
        
        # 处理掩码
        if attention_mask.all():
            attention_mask = None

        # Transformer层前向传播
        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x, 
                self_attn_padding_mask=attention_mask,
                need_head_weights=need_head_weights
            )
            
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            
            if need_head_weights:
                attn_weights.append(attn.transpose(1, 0))
        
        # 层归一化和维度转换回 (B, T, E)
        x1 = self.emb_layer_norm_after(x)
        x1 = x1.transpose(0, 1)
        
        # 最后的隐藏表示
        if len(repr_layers) > 0 and (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x1
        
        # 语言模型输出
        logits = self.lm_head(x1)
        
        has_nan = torch.isnan(logits).any()

        if has_nan:
            print("jkfdshjfk")


        # 构建输出 
        result = {
            "logits": logits,
            "representations": hidden_representations
        }
        
        if need_head_weights:
            attentions = torch.stack(attn_weights, 1)
            if attention_mask is not None:
                attention_mask = 1 - attention_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
        
        return result
    
    def embed_sequence(self, tokens, repr_layers=None, average=True):
        """与原始CaLM API兼容的方法"""
        if repr_layers is None:
            repr_layers = [self.config.num_layers]
        
        with torch.no_grad():
            output = self.forward(
                input_ids=tokens,
                repr_layers=repr_layers
            )
        
        repr_ = output["representations"][repr_layers[0]]
        if average:
            return repr_.mean(axis=1)
        else:
            return repr_

class CaLMTokenizer(PreTrainedTokenizer):
    def __init__(
        self, 
        vocab_file=None, 
        use_codons=True, 
        **kwargs
    ):
          
        # 根据架构选择词汇表
        architecture = "CodonModel" if use_codons else "ESM-1b"
        alphabet = Alphabet.from_architecture(architecture)
        
        # 初始化词汇表
        vocab = {tok: idx for idx, tok in enumerate(alphabet.all_toks)}
        
        # 特殊token
        special_tokens = {
            "unk_token": "<unk>",
            "sep_token": "<sep>",
            "pad_token": "<pad>",
            "cls_token": "<cls>",
            "mask_token": "<mask>",
            "eos_token": "<eos>"
        }

        # 更新kwargs中的特殊token
        for token_type, token_value in special_tokens.items():
            if token_type not in kwargs:
                kwargs[token_type] = token_value

        # 设置词汇表
        self._vocab = vocab
        self.alphabet = alphabet
        self._use_codons = use_codons

        # 添加特殊token到词汇表
        for token_type, token_value in special_tokens.items():
            if token_value not in self._vocab:
                self._vocab[token_value] = len(self._vocab)


        # 调用父类初始化
        super().__init__(
            vocab_file=vocab_file,
            **kwargs
        )

    def get_vocab(self) -> Dict[str, int]:
        """返回词汇表"""
        return dict(self._vocab)
#######################################################################################
    def _tokenize(self, text: str) -> List[str]:
        """将文本分词"""
        return self.alphabet.tokenize(text)
    
    def _convert_token_to_id(self, token: str) -> int:
        """将token转换为ID"""
        return self._vocab.get(token, self._vocab[self.unk_token])
    
    def _convert_id_to_token(self, index: int) -> str:
        """将ID转换为token"""
        # 反向词汇表
        rev_vocab = {v: k for k, v in self._vocab.items()}
        return rev_vocab.get(index, self.unk_token)
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """将tokens转换为字符串"""
        return ' '.join(tokens)
    
    def save_vocabulary(self, save_directory: str, filename_prefix: str = None) -> Tuple[str, str]:
        """保存词汇表"""
        vocab_file = os.path.join(
            save_directory, 
            f"{filename_prefix + '-' if filename_prefix else ''}vocab.json"
        )
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)
        
        return vocab_file, None
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """从预训练模型加载分词器"""
        try:
            # 尝试加载词汇表
            vocab_file = os.path.join(pretrained_model_name_or_path, 'vocab.json')
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab = json.load(f)

            # 加载分词器配置
            config_file = os.path.join(pretrained_model_name_or_path, 'tokenizer_config.json')
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 加载特殊token映射
            special_tokens_file = os.path.join(pretrained_model_name_or_path, 'special_tokens_map.json')
            with open(special_tokens_file, 'r', encoding='utf-8') as f:
                special_tokens = json.load(f)

            # 提取参数
            use_codons = config.get('use_codons', True)

            # 创建分词器实例
            tokenizer = cls(
                vocab_file=vocab_file, 
                use_codons=use_codons, 
                **special_tokens,
                **kwargs
            )
            
            # 设置词汇表
            tokenizer._vocab = vocab

            return tokenizer

        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            # 回退到默认初始化
            return cls(use_codons=True, **kwargs)

    def prepare_for_model(
        self,
        ids: List[int],
        text=None,
        text_pair=None,
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: int = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """准备模型输入"""
        self.alphabet.prepend_bos = True
        self.alphabet.append_eos = True

        # 处理输入 ids
        if isinstance(ids, list):
            # 复制列表以避免修改原始输入
            ids = ids.copy()
        elif isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        # 添加特殊 token
        if add_special_tokens:
            if self.alphabet.prepend_bos:
                ids.insert(0, self._convert_token_to_id(self.cls_token))
            if self.alphabet.append_eos:
                ids.append(self._convert_token_to_id(self.eos_token))
        
        # 处理填充
        if padding or max_length:
            # 计算需要填充的长度
            target_length = max_length if max_length else len(ids)
            
            # 填充或截断
            if len(ids) > target_length:
                # 截断
                ids = ids[:target_length]
            else:
                # 填充
                pad_token_id = self._convert_token_to_id(self.pad_token)
                ids.extend([pad_token_id] * (target_length - len(ids)))
        
        # 创建输入张量
        input_ids = torch.tensor(ids, dtype=torch.long)
        
        # 创建注意力掩码
        attention_mask = torch.tensor(
            [1 if id != self._convert_token_to_id(self.pad_token) else 0 for id in ids], 
            dtype=torch.long
        )
        
        # 返回结果
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

    def save_pretrained(
        self, 
        save_directory: str, 
        filename_prefix: str = None,
        legacy_format: bool = False,
        filename: str = None
    ):
        """保存分词器到指定目录"""
        # 确保目录存在
        os.makedirs(save_directory, exist_ok=True)

        # 保存词汇表
        vocab_file = os.path.join(
            save_directory, 
            f"{filename_prefix + '-' if filename_prefix else ''}vocab.json"
        )
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)

        # 保存分词器配置
        tokenizer_config = {
            'use_codons': self._use_codons,
            'special_tokens': {
                'unk_token': self.unk_token,
                'sep_token': self.sep_token,
                'pad_token': self.pad_token,
                'cls_token': self.cls_token,
                'mask_token': self.mask_token,
                'eos_token': self.eos_token
            }
        }
        
        config_file = os.path.join(save_directory, 'tokenizer_config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

        # 保存特殊token映射
        special_tokens_map = {
            'unk_token': self.unk_token,
            'sep_token': self.sep_token,
            'pad_token': self.pad_token,
            'cls_token': self.cls_token,
            'mask_token': self.mask_token,
            'eos_token': self.eos_token
        }
        
        special_tokens_file = os.path.join(save_directory, 'special_tokens_map.json')
        with open(special_tokens_file, 'w', encoding='utf-8') as f:
            json.dump(special_tokens_map, f, ensure_ascii=False, indent=2)

        return (vocab_file, config_file, special_tokens_file)

class ArgDict:
    def __init__(self, d):
        self.__dict__ = d

_ARGS = {
    'max_positions': 1024,
    'batch_size': 46,
    'accumulate_gradients': 40,
    'mask_proportion': 0.25,
    'leave_percent': 0.10,
    'mask_percent': 0.80,
    'warmup_steps': 1000,
    'weight_decay': 0.1,
    'lr_scheduler': 'warmup_cosine',
    'learning_rate': 4e-4,
    'num_steps': 121000,
    'num_layers': 12,
    'embed_dim': 768,
    'attention_dropout': 0.,
    'logit_bias': False,
    'rope_embedding': True,
    'ffn_embed_dim': 768*4,
    'attention_heads': 12
}
ARGS = ArgDict(_ARGS)

def convert_calm_to_huggingface():
    # 加载原始模型
    alphabet = Alphabet.from_architecture('CodonModel')
    original_model = ProteinBertModel(ARGS, alphabet)
    
    # 加载预训练权重
    with open("/hpcfs/fhome/yangchh/ai/finetuneFMs/FMs/CaLM/calm/calm_weights/calm_weights.ckpt", 'rb') as handle:
        state_dict = pickle.load(handle)
        original_model.load_state_dict(state_dict)
    
    # 创建Hugging Face配置
    config = CaLMConfig(
        max_positions=ARGS.max_positions,
        num_layers=ARGS.num_layers,
        embed_dim=ARGS.embed_dim,
        attention_heads=ARGS.attention_heads,
        ffn_embed_dim=ARGS.ffn_embed_dim,
        attention_dropout=ARGS.attention_dropout,
        logit_bias=ARGS.logit_bias,
        rope_embedding=ARGS.rope_embedding,
        vocab_size=len(alphabet),
        pad_token_id=alphabet.padding_idx,
        mask_token_id=alphabet.mask_idx,
        cls_token_id=alphabet.cls_idx,
        eos_token_id=alphabet.eos_idx,
    )
    
    # 创建Hugging Face模型
    hf_model = CaLMModel(config)
    
    # 映射和复制权重
    # 这里可能需要根据具体的模型架构进行适配
    # 简单情况下，如果键名相同或对应关系明确，可以直接复制
    hf_state_dict = {}
    for key, value in original_model.state_dict().items():
        # 可能需要重命名键
        hf_key = key
        hf_state_dict[hf_key] = value.clone()
    
    # 加载映射后的权重
    missing_keys, unexpected_keys = hf_model.load_state_dict(hf_state_dict, strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    
    config.model_type = "bert"  # 或其他 Transformers 识别的类型

    # 保存模型
    hf_model.save_pretrained("huggingface_calm")
    config.save_pretrained("huggingface_calm")
    

    # 保存分词器
    tokenizer = CaLMTokenizer(use_codons=True)
    tokenizer.save_pretrained("huggingface_calm")

    print("Conversion completed. Model saved to 'huggingface_calm' directory.")

# 添加一个从保存的模型加载的函数
def load_huggingface_calm():
    save_directory = "huggingface_calm"
    
    # 加载配置
    config = CaLMConfig.from_pretrained(save_directory)
    
    # 加载模型
    model = CaLMModel.from_pretrained(save_directory)
    
    # 加载分词器
    tokenizer = CaLMTokenizer.from_pretrained(save_directory)
    
    return config, model, tokenizer



if __name__ == "__main__":
    convert_calm_to_huggingface()
