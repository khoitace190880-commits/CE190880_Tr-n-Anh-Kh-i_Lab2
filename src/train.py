import argparse, json, os, random, time, copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import build_model

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_transforms(model_name):
    if model_name == "cnn":
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        eval_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        ])
        eval_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        ])
    return train_tf, eval_tf

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', default='data/fer2013', help='Folder containing train/ and test/')
    ap.add_argument('--model', choices=['cnn','transfer'], required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--freeze-backbone', action='store_true', help='Train only ResNet classifier head')
    ap.add_argument('--no-pretrained', action='store_true', help='Do not download/use pretrained ResNet weights')
    args=ap.parse_args()
    seed_everything(args.seed)
    data=Path(args.data)
    train_dir=data/'train'; test_dir=data/'test'
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise SystemExit(f"Expected {train_dir} and {test_dir}. Extract fer2013.zip into data/fer2013/")
    train_tf, eval_tf=get_transforms(args.model)
    base=datasets.ImageFolder(str(train_dir))
    test_ds=datasets.ImageFolder(str(test_dir), transform=eval_tf)
    classes=base.classes
    if test_ds.classes != classes:
        raise SystemExit(f"Train/test class folders differ: {classes} vs {test_ds.classes}")
    y=np.array(base.targets); idx=np.arange(len(y))
    train_idx,val_idx=train_test_split(idx,test_size=0.10,random_state=args.seed,stratify=y)
    train_full=datasets.ImageFolder(str(train_dir),transform=train_tf)
    val_full=datasets.ImageFolder(str(train_dir),transform=eval_tf)
    train_ds=Subset(train_full,train_idx); val_ds=Subset(val_full,val_idx)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pin=device.type=='cuda'
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=pin)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=pin)
    test_loader=DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=pin)
    model=build_model(args.model,pretrained=(not args.no_pretrained and args.model=='transfer'),freeze_backbone=args.freeze_backbone).to(device)
    counts=np.bincount(y,minlength=len(classes)).astype(np.float32)
    class_weights=(len(y)/(len(classes)*np.maximum(counts,1)))
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(class_weights,dtype=torch.float32,device=device))
    lr=args.lr if args.lr is not None else (1e-3 if args.model=='cnn' else 1e-4)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=lr,weight_decay=1e-4)
    out=Path('results')/args.model; out.mkdir(parents=True,exist_ok=True)
    weights_dir=Path('weights'); weights_dir.mkdir(exist_ok=True)
    history={'train_loss':[],'val_loss':[],'train_accuracy':[],'val_accuracy':[]}
    best_f1=-1; best_state=None
    for epoch in range(args.epochs):
        start=time.time()
        model.train(); loss_sum=0; correct=0; total=0
        for x,yb in train_loader:
            x=x.to(device); yb=yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(x); loss=criterion(logits,yb); loss.backward()
            optimizer.step()
            loss_sum += loss.item()*yb.size(0); correct += (logits.argmax(1)==yb).sum().item(); total += yb.size(0)
        tr_loss=loss_sum/max(total,1); tr_acc=correct/max(total,1)
        model.eval(); vl_sum=0; vy=[]; vp=[]
        with torch.no_grad():
            for x,yb in val_loader:
                x=x.to(device); yb=yb.to(device)
                logits=model(x); loss=criterion(logits,yb)
                vl_sum += loss.item()*yb.size(0)
                vp.extend(logits.argmax(1).cpu().numpy()); vy.extend(yb.cpu().numpy())
        va_loss=vl_sum/max(len(vy),1); va_acc=accuracy_score(vy,vp); va_f1=f1_score(vy,vp,average='macro',zero_division=0)
        for k,v in [('train_loss',tr_loss),('val_loss',va_loss),('train_accuracy',tr_acc),('val_accuracy',va_acc)]: history[k].append(float(v))
        print(f"Epoch {epoch+1:02d}/{args.epochs} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f} val_macro_f1={va_f1:.4f} time={time.time()-start:.1f}s",flush=True)
        if va_f1>best_f1:
            best_f1=va_f1; best_state=copy.deepcopy(model.state_dict())
            torch.save({'model':best_state,'model_name':args.model,'classes':classes,'class_to_idx':base.class_to_idx,'history':history},weights_dir/f'{args.model}_best.pth')
    if best_state is not None: model.load_state_dict(best_state)
    model.eval(); ys=[]; ps=[]; probs=[]; paths=[]
    with torch.no_grad():
        for x,yb in test_loader:
            x=x.to(device)
            logits=model(x); pr=torch.softmax(logits,dim=1)
            ps.extend(logits.argmax(1).cpu().numpy()); probs.extend(pr.max(1).values.cpu().numpy()); ys.extend(yb.numpy())
    test_acc=accuracy_score(ys,ps); test_f1=f1_score(ys,ps,average='macro',zero_division=0)
    cm=confusion_matrix(ys,ps,labels=list(range(len(classes))))
    report=classification_report(ys,ps,target_names=classes,output_dict=True,zero_division=0)
    # Approximate throughput measured on test loader; report both model inference throughput and end-to-end not included.
    dummy=next(iter(test_loader))[0][:min(args.batch_size,32)].to(device)
    with torch.no_grad():
        for _ in range(5): model(dummy)
        if device.type=='cuda': torch.cuda.synchronize()
        t0=time.perf_counter()
        n=20
        for _ in range(n): model(dummy)
        if device.type=='cuda': torch.cuda.synchronize()
        fps=(dummy.size(0)*n)/(time.perf_counter()-t0)
    params=sum(p.numel() for p in model.parameters())
    result={'model':args.model,'device':str(device),'classes':classes,'train_images':len(train_idx),'validation_images':len(val_idx),'test_images':len(test_ds),'test_accuracy':float(test_acc),'test_macro_f1':float(test_f1),'parameters':int(params),'inference_fps_batch_estimate':float(fps),'best_validation_macro_f1':float(best_f1),'classification_report':report,'confusion_matrix':cm.tolist(),'class_counts_train':dict(zip(classes,counts.astype(int).tolist()))}
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    (out/'classification_report.txt').write_text(classification_report(ys,ps,target_names=classes,zero_division=0),encoding='utf-8')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    axs[0].plot(history['train_loss'],label='Train'); axs[0].plot(history['val_loss'],label='Validation'); axs[0].set_title('Loss'); axs[0].set_xlabel('Epoch'); axs[0].legend()
    axs[1].plot(history['train_accuracy'],label='Train'); axs[1].plot(history['val_accuracy'],label='Validation'); axs[1].set_title('Accuracy'); axs[1].set_xlabel('Epoch'); axs[1].legend()
    fig.tight_layout(); fig.savefig(out/'training_curves.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(cm,cmap='Blues'); ax.set_xticks(range(len(classes)),classes,rotation=45,ha='right'); ax.set_yticks(range(len(classes)),classes); ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f"Confusion matrix - {args.model}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(out/'confusion_matrix.png',dpi=160); plt.close(fig)
    # Save at least 10 incorrect examples for the required error analysis.
    wrong=[]
    for i,(yt,yp) in enumerate(zip(ys,ps)):
        if yt!=yp: wrong.append((i,yt,yp,float(probs[i])))
    with open(out/'misclassified_examples.csv','w',encoding='utf-8') as f:
        f.write('test_index,true_label,predicted_label,confidence\n')
        for i,yt,yp,cf in wrong[:100]: f.write(f'{i},{classes[yt]},{classes[yp]},{cf:.6f}\n')
    # Export first 20 errors as images with true/pred labels.
    from PIL import Image, ImageDraw
    samples=[]
    test_paths=[p for p,_ in test_ds.samples]
    for i,yt,yp,cf in wrong[:20]:
        im=Image.open(test_paths[i]).convert('RGB').resize((192,192))
        canvas=Image.new('RGB',(192,224),'white'); canvas.paste(im,(0,0))
        ImageDraw.Draw(canvas).text((4,196),f'T:{classes[yt]} P:{classes[yp]} {cf:.2f}',fill='black')
        samples.append(canvas)
    if samples:
        cols=4; rows=(len(samples)+cols-1)//cols
        sheet=Image.new('RGB',(cols*192,rows*224),'white')
        for i,im in enumerate(samples): sheet.paste(im,((i%cols)*192,(i//cols)*224))
        sheet.save(out/'misclassified_grid.jpg',quality=90)
    print(json.dumps({k:result[k] for k in ['model','device','test_accuracy','test_macro_f1','parameters','inference_fps_batch_estimate']},indent=2))
    print(f"Saved outputs to {out.resolve()} and weights/{args.model}_best.pth")

if __name__=='__main__': main()
