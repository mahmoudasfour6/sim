import json, numpy as np
from scipy.ndimage import median_filter
d=json.load(open('raw.json'))
T={'black':298.15,'red':323.15,'green':348.15,'magenta':373.15,'blue':398.15}
rows=[]; rep={}
for n,v in d.items():
    V=np.array(v['V']); L=np.array(v['log10J'])
    o=np.argsort(V); V,L=V[o],L[o]
    keep=(V>0.02)&(V<1.47)
    V,L=V[keep],L[keep]
    for _ in range(3):                                   # drop isolated JPEG/edge artefacts
        med=median_filter(L,size=9,mode='nearest')
        ok=np.abs(L-med)<0.25
        V,L=V[ok],L[ok]
    grid=np.round(np.arange(0.04,1.46,0.01),3)
    out=[]
    for g in grid:
        s=np.abs(V-g)<=0.005
        if s.sum()>=1:
            out.append((g,float(np.median(L[s]))))
    out=np.array(out)
    # monotonic sanity outside the noise floor
    hi=out[:,1]>-8
    nonmono=int(np.sum(np.diff(out[hi,1])< -0.05))
    rep[n]={'T_K':T[n],'points':len(out),'nonmonotone_steps':nonmono,'logJ_range':[out[:,1].min(),out[:,1].max()]}
    for g,l in out:
        rows.append((T[n],g,10**l))
import csv
with open('iv.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['T_K','V','I_A'])
    for r in rows: w.writerow([f'{r[0]:.2f}',f'{r[1]:.3f}',f'{r[2]:.4e}'])
print(json.dumps(rep,indent=0))
