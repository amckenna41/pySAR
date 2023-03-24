################################################################################
#################                    Plots                     #################
################################################################################

import matplotlib.pyplot as plt
import seaborn as sns
import os

from .globals_ import OUTPUT_FOLDER, CURRENT_DATETIME

def plot_reg(Y_true, Y_pred, r2, output_folder="", show_plot=False):
    """
    Plot regression plot of observed (Y_true) vs predicted activity values (Y_pred).

    Parameters
    ----------
    :Y_true : np.ndarray
        array of observed values.
    :Y_pred : np.ndarray
        array of predicted values.
    :r2 : float
        r2 score value.
    :output_folder : str (default="")
        output folder to store regression plot to, if empty input it will be stored in 
        the OUTPUT_FOLDER global var.
    :show_plot : bool (default=False)
        whether to display plot or not when function is run, if False the plot is just
        saved to output folder. 

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax = sns.regplot(x=Y_true, y=Y_pred, marker="+", truncate=False, fit_reg=True)
    r2_annotation = 'R2: {:.3f} '.format(r2)
    ax.text(0.15, 0.92, r2_annotation, ha="left", va="top", fontsize=15, color="green",
        fontweight="bold", transform=ax.transAxes)
    plt.xlabel('Predicted Value', fontdict=dict(weight='bold'), fontsize=12)
    plt.ylabel('Observed Value', fontdict=dict(weight='bold'), fontsize=12)
    plt.title('Observed vs Predicted values for protein activity',fontdict=dict(weight='bold'), fontsize=15)

    #set output folder to default if input param empty or None
    if (output_folder == "" or output_folder == None):
        output_folder = OUTPUT_FOLDER
    else:
        output_folder = output_folder + "_" + CURRENT_DATETIME
    
    #create output folder if it doesnt exist
    if not (os.path.isdir(output_folder)):
        os.makedirs(output_folder)    

    plt.savefig(os.path.join(output_folder, 'model_regPlot.png'))  #save plot to output folder
    if (show_plot): #display plot   
        plt.show(block=False)
        plt.pause(3)
        plt.close()