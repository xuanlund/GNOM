"""Final thin-line ROC and precision-recall plates for both experiments."""
from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"figures"/"performance"
mpl.rcParams.update({"font.family":"Arial","font.size":8,"axes.linewidth":.8,
 "xtick.major.width":.8,"ytick.major.width":.8,"xtick.direction":"out","ytick.direction":"out",
 "svg.fonttype":"none","pdf.fonttype":42,"savefig.facecolor":"white"})

def plate(df,models,colors,title,stem):
    y=df.label.to_numpy();fig,axs=plt.subplots(1,2,figsize=(7.15,3.15))
    for m in models:
        p=df[m].to_numpy();fpr,tpr,_=roc_curve(y,p);precision,recall,_=precision_recall_curve(y,p)
        focal=m in ("GNOM","Full GNOM");lw=1.30 if focal else 1.05;z=5 if focal else 3
        axs[0].plot(fpr,tpr,color=colors[m],lw=lw,label=f"{m} (AUC = {roc_auc_score(y,p):.3f})",zorder=z)
        axs[1].plot(recall,precision,color=colors[m],lw=lw,label=f"{m} (AP = {average_precision_score(y,p):.3f})",zorder=z)
    axs[0].plot([0,1],[0,1],color="#A9AEB3",lw=.75,zorder=1)
    axs[1].axhline(y.mean(),color="#A9AEB3",lw=.75,zorder=1)
    for i,ax in enumerate(axs):
        ax.set(xlim=(0,1),ylim=(0,1.01),xticks=[0,.2,.4,.6,.8,1],yticks=[0,.2,.4,.6,.8,1])
        ax.grid(True,color="#D8DDE3",lw=.48,alpha=.78);ax.spines[["top","right"]].set_visible(True)
        ax.legend(loc="lower right",frameon=True,facecolor="white",framealpha=.94,edgecolor="#B5BAC0",fontsize=7)
        ax.text(-.15,1.035,"ab"[i],transform=ax.transAxes,fontsize=11,fontweight="bold")
    axs[0].set(xlabel="False-positive rate",ylabel="True-positive rate",title="ROC")
    axs[1].set(xlabel="Recall",ylabel="Precision",title="Precision–recall")
    fig.suptitle(title,y=.995,fontsize=10,fontweight="bold")
    fig.subplots_adjust(left=.09,right=.985,bottom=.16,top=.84,wspace=.28)
    fig.savefig(OUT/f"{stem}.png",dpi=600,bbox_inches="tight",facecolor="white")
    fig.savefig(OUT/f"{stem}.svg",bbox_inches="tight",facecolor="white");plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    c=pd.read_csv(ROOT/"predictions"/"comparison_oof_predictions.csv",encoding="utf-8-sig")
    cm=["MLP","GWR","GNNWLR","GNOM"];cc={"MLP":"#B9BEC5","GWR":"#77B7D5","GNNWLR":"#376F91","GNOM":"#D95F02"}
    plate(c,cm,cc,"Model comparison","comparison_roc_pr_consistent")
    a=pd.read_csv(ROOT/"predictions"/"ablation_oof_predictions.csv",encoding="utf-8-sig")
    am=["Local only","FNO + local","Full GNOM"];ac={"Local only":"#B9BEC5","FNO + local":"#67B7E1","Full GNOM":"#E1812C"}
    plate(a,am,ac,"Progressive ablation","ablation_roc_pr_consistent")
if __name__=="__main__":main()
