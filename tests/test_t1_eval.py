import torch, numpy as np, math
from transformers import Qwen2Config, Qwen2ForCausalLM
from pa4 import t1_eval as E
torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=1000, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)
m = Qwen2ForCausalLM(cfg).eval()
wins=[]
for L in (513, 300, 513, 2):
    ids=torch.randint(0,1000,(L,)).tolist(); wins.append(dict(token_ids=ids,score_target_offsets=[1,L],scored_tokens=L-1))
s,nll,cnt=E.perplexity(m,wins,bs=3,device='cpu'); print(s, cnt)
# reference: unbatched
tot=0;n=0
for w in wins:
    ids=torch.tensor([w['token_ids']]); lg=m(input_ids=ids).logits
    l=torch.nn.functional.cross_entropy(lg[0,:-1],ids[0,1:],reduction='sum'); tot+=l.item(); n+=ids.shape[1]-1
print('ref', tot/n, 'ours', s['mean_nll']); assert abs(tot/n-s['mean_nll'])<1e-4
reqs=[dict(ctx=torch.randint(0,1000,(20+i,)).tolist(), conts=[torch.randint(0,1000,(3+j,)).tolist() for j in range(4)], char_len=[10,12,9,20], gold=i%4) for i in range(5)]
st,ll,acc,accn=E.hellaswag_eval(m,reqs,bs=7,device='cpu')
# reference
for i,r in enumerate(reqs):
    for j,c in enumerate(r['conts']):
        ids=torch.tensor([r['ctx']+c]); lp=torch.log_softmax(m(input_ids=ids).logits[0],-1)
        ref=sum(lp[len(r['ctx'])-1+k, c[k]].item() for k in range(len(c)))
        assert abs(ref-ll[i,j])<1e-3,(ref,ll[i,j])
print('hs ok',st)
