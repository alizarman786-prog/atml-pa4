import torch, math
import pa4.t2c_kv as C
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache
dev='cpu'
# monkeypatch cuda device strings -> cpu for local test
import re
torch.manual_seed(0)
cfg=Qwen2Config(vocab_size=500,hidden_size=256,intermediate_size=512,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=1024)
m=Qwen2ForCausalLM(cfg).eval()
# pack roundtrip
K=torch.randn(1,2,200,128).half(); V=torch.randn(1,2,200,128).half()
for lay in ('kivi','token'):
    P=C.pack_layer(K,V,4,lay); K2,V2=C.unpack_layer(P)
    Lq=192
    refK=(C.fake_quant_channelwise if lay=='kivi' else C.fake_quant_tokenwise)(K[:,:,:Lq],4)
    assert torch.equal(K2[:,:,:Lq],refK) and torch.equal(K2[:,:,Lq:],K[:,:,Lq:])
    assert torch.equal(V2[:,:,:Lq],C.fake_quant_tokenwise(V[:,:,:Lq],4))
    print(lay,'serialized',C.serialized_bytes(P),'formula per (1 layer,2 heads)', 2*2*128*((200//64)*(32+4)+2*(200%64)))
    print('relerr K', ((K2-K).float().norm()/K.float().norm()).item())
geo=C.Geo(type('c',(),dict(num_hidden_layers=28,num_key_value_heads=2,hidden_size=1536,num_attention_heads=12))())
for L in (2000,2100,4100):
    b=geo.Mb(L,4); n1=geo.evict_n(b,False); n2=geo.evict_n(b,True); nh=geo.hybrid_n(L,b)
    print(L,'budget',b,'frac',b/geo.M16(L),'evict n',n1,n2,'hybrid n',nh, geo.hybrid_bytes(L,nh)<=b, geo.hybrid_bytes(L,nh+1)>b)
C.DEV='cpu'
ids=torch.randint(0,500,(300,)).tolist(); L=260
full=m(input_ids=torch.tensor([ids])).logits[0]
cache,_=C.prefill(m,ids[:L])
lg=C.fwd(m,ids[L:],cache,L,L,logits_all=True)
print('split vs full',(lg-full[L:]).abs().max().item())
sel=torch.tensor(sorted(set(range(8))|set(range(L-64,L))|set(range(100,L,3))))
cache3,_=C.prefill(m,ids[:L])
kv3=[(k[:,:,sel].contiguous(),v[:,:,sel].contiguous()) for k,v in C.cache_kv(cache3)]
ev=C.fwd(m,ids[L:],C.cache_from(kv3),L,len(sel),logits_all=True)
T=len(ids); mm=torch.tril(torch.ones(T,T,dtype=torch.bool)); drop=torch.ones(L,dtype=torch.bool); drop[sel]=False; mm[L:,:L]&=~drop[None]
mask4=torch.zeros(1,1,T,T).masked_fill(~mm,torch.finfo(torch.float32).min)
ref=m(input_ids=torch.tensor([ids]),attention_mask=mask4).logits[0,L:]
print('evict vs 4D mask',(ev-ref).abs().max().item())
# wrong positions would differ:
ev_bad=C.fwd(m,ids[L:],C.cache_from([(k[:,:,sel].contiguous(),v[:,:,sel].contiguous()) for k,v in C.cache_kv(C.prefill(m,ids[:L])[0])]),len(sel),len(sel),logits_all=True)
print('naive renumbered positions diff',(ev_bad-ref).abs().max().item())
# question scores + policies
cache4,_=C.prefill(m,ids[:L]); sc=C.question_scores(m,C.clone_cache(cache4),ids[L:],L); print('scores',sc.shape, sc.sum(1))
assert cache4.get_seq_length()==L
kvf=C.cache_kv(cache4)
class Tok: eos_token_id=1
Tok.convert_tokens_to_ids=staticmethod(lambda x:2)
for pol in C.POLICIES:
    kv,sel,maint=C.apply_policy(pol,kvf,L,None,sc,n=100)
    toks,first,tm=C.greedy(m,Tok,C.cache_from([(k.clone(),v.clone()) for k,v in kv]),ids[L:],L,maint,max_new=70)
    print(pol,len(kv[0][0][0,0]), toks[:5], maint.qend if maint else None)
