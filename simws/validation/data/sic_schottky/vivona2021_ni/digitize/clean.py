import json, numpy as np, csv
from scipy.ndimage import median_filter
d=json.load(open('raw.json'))
T={f'T{c}':273.15+c for c in (25,40,55,70,85,100,115)}
rows=[]
for n,v in d.items():
    V=np.array(v['V']); L=np.array(v['log10J']); o=np.argsort(V); V,L=V[o],L[o]
    for _ in range(3):
        med=median_filter(L,size=9,mode='nearest'); ok=np.abs(L-med)<0.25; V,L=V[ok],L[ok]
    for g in np.round(np.arange(0.04,0.97,0.01),3):
        s=np.abs(V-g)<=0.005
        if s.sum(): rows.append((T[n],g,10**float(np.median(L[s]))))
with open('iv.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['T_K','V','I_A'])
    for r in rows: w.writerow([f'{r[0]:.2f}',f'{r[1]:.3f}',f'{r[2]:.4e}'])
print(len(rows))
