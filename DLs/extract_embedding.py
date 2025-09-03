import os 
import torch
from torch.utils.data import random_split
import pandas as pd
import pandas as pd
import numpy as np
import os, sys, subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import metrics
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Dict, Any, Optional
import logging
from torch.utils.data import DataLoader, Dataset, random_split
from config import *
from plm_data_processor import PLMs_DataProcessor, Load_PLMs
# 配置日志记录 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__) 

import random
def seed_everything(seed=42):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EmbeddingExtractor:
    """模型评估处理器"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_config = self._get_model_config()

        classifier = Load_PLMs(model_config)
        self.model, self.tokenizer = classifier.model, classifier.tokenizer
        self.processor = PLMs_DataProcessor(self.tokenizer, model_config)

        self.data_df = None
        self.dataset = None
        self.dataloader = None

    def _get_model_config(self) -> ModelConfig:
        
        return ModelConfig(
            pretrained_model = self.config['pretrained_model'],
            max_seq_length = self.config['max_seq_length'],
            task = self.config['task'],
            pretrained_model_type = self.config["pretrained_model_type"]
        )
    
    def _load_finetune_model(self) -> None:
        """Load pretrained model with adapters."""
        try:
            # 初始化模型和分词器，根据模型名称选择加载方式
            non_frozen_params = torch.load(self.config["finetuned_params_path"])

            # # # # Assign the non-frozen parameters to the corresponding parameters of the model
            for param_name, param in self.model.named_parameters():
                if param_name in non_frozen_params:
                    param.data = non_frozen_params[param_name].data

            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _load_foundation_model(self)-> None:
        """Load pretrained model with adapters."""
        try: 
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _do_embedding(self):
        
        embedding_dict = {}
        embedding_repr_list = []
        labels_list = []
        id_list = []
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Extracting embeddings"):
                
                if "calm" in self.config['pretrained_model'].lower():
                    result = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        repr_layers=[12]
                        )
                    if self.config["task"] == "do_finetune":
                        repr_ = result["hidden_states"]
                    else:
                        repr_ = result["representations"][12]
                    embedding_repr = repr_.mean(axis=1).cpu().numpy()
                    # embedding_repr = repr_.cpu().numpy() 
                else:
                    result = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        output_hidden_states=True
                    )
                    if self.config["task"] == "do_finetune":
                        embedding_repr = result["hidden_states"][-1].detach()
                        embedding_repr = embedding_repr.mean(1).cpu().numpy()
                        # embedding_repr = embedding_repr.cpu().numpy()
                    else:
                        embedding_repr = result.last_hidden_state.detach()
                        embedding_repr = embedding_repr.mean(1).cpu().numpy()
                        # embedding_repr = embedding_repr.cpu().numpy()
                
                for emb in embedding_repr:
                    embedding_repr_list.append(emb.tolist())

                labels = batch["labels"].cpu().numpy()
                labels_list.extend(labels)

                ids = batch["id"]
                id_list.extend(ids)
        
        embedding_dict = {
            "id" : id_list,
            "label" : labels_list,
            "embedding" : embedding_repr_list
        }

        df = pd.DataFrame( embedding_dict  )

        return df

    def _run(self):
        
        self.data_df = self.processor._load_data( os.path.join(self.config["data_path"], self.config["file_name"] ) )

        self.dataset = self.processor._create_dataset(self.data_df)
        
        self.dataloader = DataLoader( self.dataset.with_format("torch", device=self.device),
                                 batch_size=self.config['batch_size'],
                                 shuffle=False,
                                 num_workers=0) 

        if self.config["task"] == "do_finetune":
            self._load_finetune_model()
        elif self.config["task"] == "do_embedding":
            self._load_foundation_model()
        
        df = self._do_embedding()

        return df

# # 设置全局随机种子
# GLOBAL_SEED = 42
# seed_everything(GLOBAL_SEED)


