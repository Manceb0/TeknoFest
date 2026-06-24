import os, json, time, glob, cv2
from ultralytics import YOLO
BACKEND=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT=os.path.join(BACKEND,"runs","focused_test"); os.makedirs(OUT,exist_ok=True)
m=YOLO(os.path.join(BACKEND,"runs","behavior_focused","weights","best.pt"))
r=m.val(split="test",project=os.path.join(BACKEND,"runs"),name="focused_test",exist_ok=True,plots=True,verbose=False,device=0,workers=0).box
f1=2*r.mp*r.mr/(r.mp+r.mr+1e-9); names=m.names
pc=[]
for i,ci in enumerate(r.ap_class_index):
    cf1=2*r.p[i]*r.r[i]/(r.p[i]+r.r[i]+1e-9)
    pc.append({"class":names[int(ci)],"P":round(float(r.p[i]),3),"R":round(float(r.r[i]),3),"F1":round(float(cf1),3),"mAP50":round(float(r.ap50[i]),3)})
imgs=[cv2.imread(f) for f in glob.glob(os.path.join(BACKEND,"..","tmp","behavior_ds","test","images","*.jpg"))[:120]]
m.predict(imgs[0],imgsz=512,device=0,verbose=False)
t=time.perf_counter()
for im in imgs: m.predict(im,imgsz=512,device=0,verbose=False)
fps=len(imgs)/(time.perf_counter()-t)
s={"overall":{"precision":round(float(r.mp),3),"recall":round(float(r.mr),3),"F1":round(float(f1),3),"mAP50":round(float(r.map50),3),"mAP50_95":round(float(r.map),3)},"per_class":pc,"fps_gpu_512":round(fps,1)}
json.dump(s,open(os.path.join(OUT,"eval_summary.json"),"w"),indent=2)
print("EVAL_DONE",json.dumps(s))
