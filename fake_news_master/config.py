from transformers import AutoModel,BertTokenizerFast
import torch 

DEVICE = "cpu"
MAX_LEN = 270
MODEL_PATH = "c3_new_model_weights2.pt"
TOKENIZER = BertTokenizerFast.from_pretrained('bert-base-uncased')
BERT = AutoModel.from_pretrained('bert-base-uncased',torch_dtype=torch.float32, 
        resume_download=True,
        cache_dir='.cache/open-llama-13b-open-instruct')