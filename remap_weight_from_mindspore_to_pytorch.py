import pickle
import torch

from collections import OrderedDict

src_path = '/mnt/qinziwei/wukong_vit_b_32_clip.pkl'
dst_path = '/mnt/qinziwei/wukong_vit_b_32_clip.pt'

    # embed_dim = state_dict["text_projection"].shape[1]
    # context_length = state_dict["positional_embedding"].shape[0]
    # vocab_size = state_dict["token_embedding.weight"].shape[0]
    # transformer_width = state_dict["ln_final.weight"].shape[0]

key_mapping = {
    'transformer.text_projection':'text_projection',
    'transformer.positional_embedding':'positional_embedding',
    'transformer.token_embedding.weight':'token_embedding.weight',
    'transformer.ln_final.weight':'ln_final.weight',
    'transformer.ln_final.bias':'ln_final.bias',
    'loss.logit_scale':'logit_scale'
}

source = pickle.load(open(src_path, 'rb'))
for s, d in key_mapping.items():
    source[d] = source[s]
    del source[s]
for k, v in source.items():
    source[k] = torch.Tensor(v)
source = OrderedDict(source)
torch.save(source, dst_path)
#                                                                                                                         {'text_projection', 'context_length', 'vocab_size', 'token_embedding.weight', 'input_resolution', 'ln_final.bias', 'positional_embedding', 'ln_final.weight', 'logit_scale'}>>> set(b.keys()) - (set(weight.keys()) & set(b.keys()))                                                                                                                                                         {'transformer.positional_embedding', 'loss.logit_scale', 'transformer.token_embedding.weight', 'transformer.ln_final.weight', 'transformer.text_projection', 'transformer.ln_final.bias'}
