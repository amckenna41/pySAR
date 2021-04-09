
import matplotlib.pyplot as plt
import seaborn as sns
import os

from globals import OUTPUT_DIR, OUTPUT_FOLDER, DATA_DIR

def plot_reg(Y_true, Y_pred, r2):

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = sns.regplot(x=Y_true,y=Y_pred,  marker="+", truncate=False,fit_reg=True)
    r2_annotation = 'R2: {:.3f} '.format(r2)
    ax.text(0.15, 0.92, r2_annotation, ha="left", va="top", fontsize=15, color="green", fontweight="bold",transform=ax.transAxes)
    plt.xlabel('Predicted Value')
    plt.ylabel('Actual Value')
    # title = '{} Protein Activity'.format(dataset_list[0])
    plt.title('Title')
    plt.savefig(os.path.join(OUTPUT_FOLDER,'regPlot.png'))
    plt.close()

def plot_features(feature_df):

    color=['red','green','blue','orange']

    plt.figure(figsize=(24,8))
    plt.subplot(1, 4, 1)
    sns.scatterplot(x="Index + Descriptor", y="R2", data=r2_df,marker='x', markers={"size":0.5}, color=color[0],linewidth=2)


#plot showing top10 mutants and their name and predicted activity value
