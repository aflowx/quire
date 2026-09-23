"""Serial latency and accuracy through a running quire-serve on JevBench's public standard tier.

JevBench measures Speed serially on its standard+judge run; the judge tier is
not public, so the standard tier stands in. The axis shown applies the
harness's self-hosted adjustment (x2 + 0.15 s).

    JEVBENCH=/path/to/jevbench python bench/jevbench/speed_probe.py   # server on :8778
"""
import json, time, statistics, sys, urllib.request
rows=[json.loads(l) for l in open(__import__('os').environ['JEVBENCH'] + '/datasets/public/original.jsonl')]
def post(state, q):
    body=json.dumps({"state":state,"questions":{"q":q}}).encode()
    req=urllib.request.Request("http://127.0.0.1:8778/v1/systemone",data=body,headers={"content-type":"application/json"})
    t=time.perf_counter(); r=json.load(urllib.request.urlopen(req)); return time.perf_counter()-t, r
def q_of(r):
    q=r['question']; out={"type":q['type'],"instructions":q['instructions']}
    if q['type']=='score': out['criteria']=list(q['criteria'])
    else: out['criteria']=q.get('criteria')
    return out
for r in rows[:3]: post(r['state'], q_of(r))  # warm
lat=[]; toks=[]; ok=0
for r in rows:
    s, resp = post(r['state'], q_of(r)); lat.append(s); toks.append(resp['usage']['input_tokens'])
    a=resp['answers']['q']
    pred = ('yes' if a['noul']>=0.5 else 'no') if a['type']=='noul' else (a['choice'] if a['type']=='choice' else str(max(a['probabilities'],key=a['probabilities'].get)))
    ok += pred==str(r['expected'])
lat.sort()
p50=lat[len(lat)//2]; p95=lat[int(len(lat)*0.95)]
print(f"standard tier n={len(rows)}  acc {ok/len(rows):.3f}  p50 {p50*1000:.0f} ms  p95 {p95*1000:.0f} ms  mean tokens {statistics.mean(toks):.0f}")
import math
adj=lambda s: s*2+0.15; sp=lambda s: 100-20*math.log10(adj(s)/0.1)
print(f"speed axis (x2+0.15): {(sp(p50)+sp(p95))/2:.1f}")
