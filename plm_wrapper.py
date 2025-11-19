from utils import *
from transformers import AutoModel, AutoTokenizer, EsmModel, AutoModelForMaskedLM, T5Tokenizer, T5ForConditionalGeneration, AutoConfig, T5EncoderModel, EsmForMaskedLM
from typing import List, Optional, Tuple, Union

MODELS = {
    'prottrans-half': 'Rostlab/prot_t5_xl_half_uniref50-enc',
    'prottrans': 'Rostlab/prot_t5_xl_uniref50',
    'protbert': 'Rostlab/prot_bert',
    'esm2-35M': 'facebook/esm2_t12_35M_UR50D',
    'esm2-150M': 'facebook/esm2_t30_150M_UR50D',
    'esm2-650M': 'facebook/esm2_t33_650M_UR50D'
}

class PLMWrapper(nn.Module):
    def __init__(self, plm, **kwargs):
        super(PLMWrapper, self).__init__()
        self.model_name = plm
        if 'prottrans' in plm:
            self.plm = T5ForConditionalGeneration.from_pretrained(MODELS[plm])
            self.tokenizer = T5Tokenizer.from_pretrained(MODELS[plm])
        else:
            self.plm = EsmForMaskedLM.from_pretrained(MODELS[plm])
            self.tokenizer = AutoTokenizer.from_pretrained(MODELS[plm])
        self.hidden_size = self.plm.config.hidden_size
        self.plm.gradient_checkpointing_enable()
        if kwargs.get('freeze_backbone', True):
            self.plm.eval()
            freeze_module(self.plm)

    # Take input sequence, return input ids and attention_mask
    def tokenize(self, input_seqs):
        input_seqs = [' '.join(s) for s in input_seqs]
        tokenized_seqs = self.tokenizer.batch_encode_plus(
            input_seqs, add_special_tokens=True, 
            padding="longest", return_tensors='pt'
        )
        input_ids, attn_mask = tokenized_seqs['input_ids'], tokenized_seqs['attention_mask']
        return input_ids.to(self.plm.device), attn_mask.to(self.plm.device)
    
    def decode(self, hidden_state):
        logits = self.plm.lm_head(hidden_state * (self.plm.model_dim ** -0.5))
        seqs = self.tokenizer.batch_decode(logits.argmax(dim=-1), skip_special_tokens=True)
        return [s.replace(' ', '') for s in seqs]

    def forward_small_batch(self, input_ids, attn_mask, **kwargs):
        outputs = self.plm(input_ids, attn_mask, labels=input_ids, output_hidden_states=True, return_dict=True)    
        hs = outputs.encoder_hidden_states if 'prottrans' in self.model_name else outputs.hidden_states
        retrieve_hs = kwargs.get('num_hidden_states', 1)
        if isinstance(retrieve_hs, list):
            embeddings = [hs[i][:, 0 if 'prottrans' in self.model_name else 1:].to('cpu') for i in retrieve_hs]
            embeddings = torch.cat(embeddings, dim=-1)
        else:
            assert isinstance(retrieve_hs, int), 'num_hidden_states must be either list or int'
            embeddings = [hs[i][:, 0 if 'prottrans' in self.model_name else 1:].to('cpu') for i in range(len(hs) - kwargs.get('num_hidden_states', 1), len(hs))]
            embeddings = torch.cat(embeddings, dim=-1)
        logits = outputs.logits.to('cpu')
        del outputs
        torch.cuda.empty_cache()
        return embeddings, logits

    def forward(self, seqs, **kwargs):
        batch = kwargs.get('batch', None)
        input_ids, attn_mask = self.tokenize(seqs) if kwargs.get('tokenize', True) else seqs
        if batch is None:
            return self.forward_small_batch(input_ids, attn_mask, **kwargs)
        else:
            enc_embeddings, logits = [], []
            for j in trange(0, len(seqs), batch):
                batch_end = min(j + batch, len(seqs))
                ij, aj, = input_ids[j: batch_end], attn_mask[j: batch_end]
                ej, lj = self.forward_small_batch(ij, aj, **kwargs)
                enc_embeddings.append(ej)
                logits.append(lj)
            enc_embeddings = torch.cat(enc_embeddings, dim=0)
            logits = torch.cat(logits, dim=0)
            return enc_embeddings, logits