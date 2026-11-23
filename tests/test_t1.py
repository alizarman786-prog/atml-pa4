import torch, numpy as np, math
from transformers import Qwen2Config, Qwen2ForCausalLM
from pa4 import t1_compress as C, t1_eval as E
from pa4.t1_run import allocate, dp_allocate
torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=1000, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True, max_position_embeddings=512)
m = Qwen2ForCausalLM(cfg).eval()
for n,p in m.named_parameters():
    if 'norm' in n: p.data.uniform_(0.5,1.5)
    if 'bias' in n: p.data.normal_(0,0.1)
print(E.check_enumeration(m))
seqs=[dict(token_ids=torch.randint(0,1000,(64,)).tolist()) for _ in range(8)]
calib,meta=C.collect_calibration(m,seqs,n_sub_tokens=128,device='cpu')
print('calib keys',len(calib), calib[(0,'down_in')]['X'].shape)
# quantizer
W=torch.randn(64,256)
for b in (2,3,4,8):
    deq,q,s,z=C.quantize_groups(W,b,return_codes=True)
    err=((deq-W)**2).mean()/(W**2).mean()
    assert q.max()<=2**b-1
    pl,mt=C.packed_bytes_matrix(64,256,b); assert len(C.pack_bits(q,b))==pl, (len(C.pack_bits(q,b)),pl)
    print('bits',b,'relerr',float(err))
# AWQ equivalence
snap=C.snapshot(m)
ids=torch.randint(0,1000,(2,64))
with torch.no_grad(): ref=m(input_ids=ids).logits
sc={4:[C.awq_search_layer(l,i,calib,4,cfg,device='cpu') for i,l in enumerate(m.model.layers)]}
for i,l in enumerate(m.model.layers): tr=C.apply_awq_scales(l,sc[4][i],cfg)
with torch.no_grad(): new=m(input_ids=ids).logits
print('awq fold max diff',float((new-ref).abs().max()), 'alphas',[{g:v['alpha'] for g,v in l.items()} for l in sc[4]])
assert torch.allclose(new,ref,atol=1e-3)
C.restore(m,snap)
with torch.no_grad(): assert torch.allclose(m(input_ids=ids).logits,ref)
# quantize with awq & error tracking; with b=8 errors small
bm={(i,p):8 for i in range(2) for p in E.PROJS}
tr=C.quantize_model(m,bm,cfg=cfg,awq={8:[C.awq_search_layer(l,i,calib,8,cfg,device='cpu') for i,l in enumerate(m.model.layers)]})
errs=C.output_errors(m,snap,calib,tr,device='cpu'); print('8bit awq max relerr',max(errs.values()))
assert max(errs.values())<1e-3
C.restore(m,snap)
bm4={(i,p):4 for i in range(2) for p in E.PROJS}
tr=C.quantize_model(m,bm4); e1=np.mean(list(C.output_errors(m,snap,calib,tr,device='cpu').values()))
C.restore(m,snap)
tr=C.quantize_model(m,bm4,cfg=cfg,awq=sc); e2=np.mean(list(C.output_errors(m,snap,calib,tr,device='cpu').values()))
print('4bit RTN err',e1,'AWQ err',e2)
C.restore(m,snap)
print(C.model_storage(m,bits_map=bm4,mode='quant'))
# pruning
st=C.apply_pruning(m,calib,'wanda',sparsity=0.5); print(st); print(C.count_nonzeros(m)); print(C.model_storage(m,mode='sparse16'))
C.restore(m,snap)
st=C.apply_pruning(m,calib,'wanda',nm=True); print('2:4',st)
W=m.model.layers[0].mlp.down_proj.weight; assert ((W.reshape(W.shape[0],-1,4)!=0).sum(-1)<=2).all()
C.restore(m,snap)
# variable per-row
k={(i,p):torch.full((l.weight.shape[0],),3,dtype=torch.long) for i,p,l in E.eligible(m)}
st=C.apply_pruning(m,calib,'magnitude',k_alloc=k); assert st['zeros']==sum(v.sum() for v in k.values()); C.restore(m,snap)
s=allocate(np.random.rand(28),0.75,0.16); print('alloc mean',s.mean(),s.min(),s.max())
s=allocate(np.random.rand(28),0.5,0.16); print('alloc mean',s.mean(),s.min(),s.max())
cost=[{b:np.random.rand()*(9-b)*(1+u%4) for b in (2,3,4,8)} for u in range(112)]
nb=[{b:(1000+u%4*30000)*b for b in (2,3,4,8)} for u in range(112)]
budget=sum(n[4] for n in nb); asg,c=dp_allocate(cost,nb,budget,quantum=512); print('dp',{b:asg.count(b) for b in (2,3,4,8)}, sum(n[a] for n,a in zip(nb,asg))<=budget, c, sum(x[4] for x in cost))
# brute-force check on a tiny instance
import itertools
cs=[{b:np.random.rand() for b in (2,3,4,8)} for _ in range(5)]; ns=[{b:(u+1)*b*100 for b in (2,3,4,8)} for u in range(5)]; B=sum(n[4] for n in ns)
bf=min((sum(c[a] for c,a in zip(cs,ch)),ch) for ch in itertools.product((2,3,4,8),repeat=5) if sum(n[a] for n,a in zip(ns,ch))<=B)
a2,v2=dp_allocate(cs,ns,B,quantum=100); print('bf',round(bf[0],6),'dp',round(v2,6)); assert abs(bf[0]-v2)<1e-9
# mixed bits with awq + only
sc={b:[C.awq_search_layer(l,i,calib,b,cfg,device='cpu') for i,l in enumerate(m.model.layers)] for b in (2,4,8)}
bm={(i,p):(8 if p in ('q_proj','k_proj','v_proj') else 2 if p=='down_proj' else 4) for i in range(2) for p in E.PROJS}
tr=C.quantize_model(m,bm,cfg=cfg,awq=sc); errs=C.output_errors(m,snap,calib,tr,device='cpu')
print({p:round(errs[(0,p)],5) for p in E.PROJS}); C.restore(m,snap)
tr=C.quantize_model(m,bm,cfg=cfg,awq=sc,blocks={1},only={'o'}); errs=C.output_errors(m,snap,calib,tr,device='cpu')
print('only o in block1:',{k:round(v,5) for k,v in errs.items() if v>1e-6}); C.restore(m,snap)
print('ALL OK')
