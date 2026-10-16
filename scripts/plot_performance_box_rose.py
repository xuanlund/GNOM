"""Matched box-and-rose plates adapted from the supplied visual template."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"figures"/"performance"
mpl.rcParams.update({"font.family":"Arial","font.size":8,"axes.linewidth":.8,
 "xtick.major.width":.8,"ytick.major.width":.8,"svg.fonttype":"none","pdf.fonttype":42,
 "savefig.facecolor":"white"})
METRICS=["OA","Precision","Recall","F1","ROC-AUC","AP"]

def vals(y,p):
    z=p>=.5
    return [accuracy_score(y,z),precision_score(y,z),recall_score(y,z),f1_score(y,z),roc_auc_score(y,p),average_precision_score(y,p)]

def fold_auc(df,models):
    return [[roc_auc_score(s.label,s[m]) for _,s in df.groupby("fold",sort=True)] for m in models]

def plate(df,models,colors,stem,rose_title):
    y=df.label.to_numpy();v=np.array([vals(y,df[m].to_numpy()) for m in models]);fa=fold_auc(df,models)
    fig=plt.figure(figsize=(7.15,3.25));ax=fig.add_subplot(121);polar=fig.add_subplot(122,polar=True)
    x=np.arange(len(models));bp=ax.boxplot(fa,positions=x,widths=.56,patch_artist=True,showfliers=False,
        medianprops={"color":"#333333","lw":1.15},whiskerprops={"color":"#333333","lw":.8},
        capprops={"color":"#333333","lw":.8})
    for b,m in zip(bp["boxes"],models):b.set(facecolor=colors[m],edgecolor="#565B60",linewidth=.8,alpha=.88)
    rng=np.random.default_rng(20260926)
    for i,(m,z) in enumerate(zip(models,fa)):
        ax.scatter(i+rng.uniform(-.045,.045,len(z)),z,s=15,facecolor=colors[m],edgecolor="#51565B",lw=.45,zorder=4)
    ax.set(xticks=x,xticklabels=models,ylabel="Five-fold ROC-AUC",ylim=(.80,.95),yticks=[.80,.85,.90,.95])
    ax.tick_params(axis="x",rotation=18);ax.spines[["top","right"]].set_visible(False)
    ax.grid(False);ax.text(-.15,1.03,"a",transform=ax.transAxes,fontsize=12,fontweight="bold")

    n=6;ang=np.arange(n)*2*np.pi/n;width=2*np.pi/n*(.68/len(models));base=.40
    for j,m in enumerate(models):
        polar.bar(ang+(j-(len(models)-1)/2)*width,v[j]-base,bottom=base,width=width,
            color=colors[m],alpha=.96,edgecolor="#9A4100" if j==len(models)-1 else "white",lw=.65,zorder=3)
    polar.set_theta_offset(np.pi/2);polar.set_theta_direction(-1);polar.set_xticks(ang);polar.set_xticklabels(METRICS)
    polar.set_ylim(.40,.92);polar.set_yticks([.50,.60,.70,.80,.90]);polar.set_yticklabels([".50",".60",".70",".80",".90"],fontsize=6.8,color="#5E646A")
    polar.set_rlabel_position(108);polar.set_facecolor("#F1F8FC");polar.grid(color="#C8D4DC",lw=.55)
    polar.spines["polar"].set_color("#687078");polar.spines["polar"].set_linewidth(.8)
    polar.set_title(rose_title,pad=14,fontweight="bold",fontsize=10)
    polar.text(-.10,1.05,"b",transform=polar.transAxes,fontsize=12,fontweight="bold")
    polar.legend(handles=[Patch(facecolor=colors[m],label=m) for m in models],loc="lower center",
        bbox_to_anchor=(.5,-.23),ncol=len(models),frameon=False,fontsize=7,handlelength=1.4,columnspacing=.9)
    fig.subplots_adjust(left=.075,right=.985,bottom=.20,top=.91,wspace=.36)
    fig.savefig(OUT/f"{stem}.png",dpi=600,bbox_inches="tight",facecolor="white")
    fig.savefig(OUT/f"{stem}.svg",bbox_inches="tight",facecolor="white");plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    c=pd.read_csv(ROOT/"predictions"/"comparison_oof_predictions.csv",encoding="utf-8-sig")
    cm=["MLP","GWR","GNNWLR","GNOM"];cc={"MLP":"#B9BEC5","GWR":"#77B7D5","GNNWLR":"#376F91","GNOM":"#D95F02"}
    plate(c,cm,cc,"comparison_box_rose_consistent","Model comparison")
    a=pd.read_csv(ROOT/"predictions"/"ablation_oof_predictions.csv",encoding="utf-8-sig")
    am=["Local only","FNO + local","Full GNOM"];ac={"Local only":"#B9BEC5","FNO + local":"#67B7E1","Full GNOM":"#E1812C"}
    plate(a,am,ac,"ablation_box_rose_consistent","Progressive ablation")
if __name__=="__main__":main()
