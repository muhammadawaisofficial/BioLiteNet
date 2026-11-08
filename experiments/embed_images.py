# Object-cropped, flip/rotation/colour-invariant MobileNetV3 embeddings used by build_folds.py.
# Usage: python embed_images.py outputs.zip dedup_embeddings.npz  (reads images under outputs/splits/ in the zip)
# Flip/rotation- and colour-invariant embeddings for near-duplicate detection.
import zipfile, io, sys, hashlib, numpy as np, torch, torchvision.models as m
from PIL import Image
zp, out = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(zp)
names = sorted(n for n in z.namelist() if n.startswith('outputs/splits/') and not n.endswith('/'))
net = m.mobilenet_v3_large(weights=m.MobileNet_V3_Large_Weights.IMAGENET1K_V1).eval()
feat = torch.nn.Sequential(net.features, net.avgpool, torch.nn.Flatten())
mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
torch.set_num_threads(12)
def d8(x):  # 8 dihedral views
    v=[x, x.flip(-1)]
    for k in (1,2,3): v += [torch.rot90(x,k,(-2,-1)), torch.rot90(x,k,(-2,-1)).flip(-1)]
    return torch.cat(v)
embs, md5s, meta = [], [], []
B=32; buf=[]
def flush():
    x=torch.stack(buf); n=len(buf)
    with torch.no_grad():
        f=feat((d8(x)-mean)/std).view(8,n,-1).mean(0)
    embs.append(torch.nn.functional.normalize(f,dim=1).numpy()); buf.clear()
for i,n in enumerate(names):
    b=z.read(n); md5s.append(hashlib.md5(b).hexdigest())
    _,_,split,cls,fn=n.split('/',4); meta.append((split,cls,fn))
    im=Image.open(io.BytesIO(b)).convert('L')
    a=np.asarray(im); ys,xs=np.where(a>25)
    if len(xs):
        x0,x1,y0,y1=xs.min(),xs.max()+1,ys.min(),ys.max()+1; s=max(x1-x0,y1-y0); cx,cy=(x0+x1)//2,(y0+y1)//2
        im=im.crop((cx-s//2-2,cy-s//2-2,cx+s//2+2,cy+s//2+2))
    im=im.resize((128,128)).convert('RGB')
    buf.append(torch.from_numpy(np.asarray(im,dtype=np.float32)/255.).permute(2,0,1))
    if len(buf)==B: flush()
    if i%1000==0: print(i,flush=True)
if buf: flush()
np.savez(out, emb=np.concatenate(embs), md5=np.array(md5s), meta=np.array(meta))
print('done', len(names))
