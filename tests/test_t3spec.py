import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
import pa4.t3_spec as S
torch.manual_seed(0)
mk=lambda h,l: Qwen2ForCausalLM(Qwen2Config(vocab_size=50,hidden_size=h,intermediate_size=2*h,num_hidden_layers=l,num_attention_heads=4,num_key_value_heads=2)).eval()
T=mk(128,2); D=mk(64,1)
prompt=[3,4,5,6,7]
for g in (1,2,4):
    r=S.check_cache_consistency(T,D,prompt,eos_ids=(),gamma=g,n=40)
    print(g,r['max_abs_prob_diff_target'],r['draft_split_prefill_max_abs_logit'],r['stats']['accepted'],r['stats']['examined'],r['stats']['rounds'],r['n_tokens'])
# draft == target -> every proposal accepted
gen=torch.Generator().manual_seed(1)
out,st,_=S.speculative_generate(T,T,prompt,4,30,gen,eos_ids=())
print('self-draft accept',st['accepted'],st['examined'],st['rounds'],len(out))
# eos stop
out,st,_=S.speculative_generate(T,D,prompt,4,200,torch.Generator().manual_seed(2),eos_ids=(7,))
print('eos',st['stop_reason'],out[-1],len(out))
out,st=S.target_generate(T,prompt,20,torch.Generator().manual_seed(3),eos_ids=())
print('target only',len(out),st['stop_reason'])
