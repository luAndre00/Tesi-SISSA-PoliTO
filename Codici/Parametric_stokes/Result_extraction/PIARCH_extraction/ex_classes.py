import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import griddata




np.set_printoptions(threshold=np.inf)
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)



area = 60 * np.ones(638)

class plot_loss:
    def all_plots(self, val):
        x = np.arange(1, 50000+1) #ascisse
        nomi = ["gamma_above","gamma_below","gamma_right","gamma_left","D_loss","mean_loss"]
        for j in range(0,6):
            y = val[:,j]
            plt.loglog(x,y)
            plt.tight_layout()
            plt.savefig(nomi[j]+'.png', transparent=True, dpi=300)
            plt.close()


#Funzione per fare lo scatter plot
class Scatter_plot:
    def plot_parameter(self, sample_points, str):
        nomi = ["gamma_above","gamma_below","gamma_right","gamma_left","D"]
        for nome in nomi:
            mu = sample_points[nome]['mu']
            thickness = 25 * np.ones(mu.shape) 
            plt.scatter(mu, s=thickness, color="r", alpha=1, marker='+') #Da sistemare
            plt.xlabel(r'$\mu$')
            plt.tight_layout()
            plt.savefig(str + '_' + nome + '.png', transparent=True, dpi=300)
            plt.close()



    def plot(self, val, x, y, str):
        xi = np.linspace(0,1,1000)
        yi = np.linspace(0,2,1000)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), val, (xi, yi), method='cubic')
        plt.imshow(zi, extent=(0, 1, 0, 2), origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(str+'.png', transparent=True, dpi=300)
        plt.close()

    #Questo serve nel caso voglia scatterare anche i punti di sample
    def plots(self, val, x, y, str, sample_points):
        xi = np.linspace(0,1,1000)
        yi = np.linspace(0,2,1000)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), val, (xi, yi), method='cubic')
        plt.imshow(zi, extent=(0, 1, 0, 2), origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar()
        #Qua devo plottare tutti i sample points, per farlo uso un ciclo for sui nomi che ho dato ai vari pezzi del dominio
        nomi = ["gamma_above","gamma_below","gamma_right","gamma_left","D"]
        if True: #cioè se voglio mettere anche i punti di sample
            for nome in nomi:
                x_sample = sample_points[nome]['x'].detach().numpy()
                y_sample = sample_points[nome]['y'].detach().numpy()

                #Questa roba serve solo per fare in modo che i grafici vengano più nitidi nei punti di training
                if nome == "D":
                    unique_x = np.unique(x_sample, return_index=True)[1]
                    x_sample = x_sample[np.sort(unique_x)]
                    unique_y = np.unique(y_sample, return_index=True)[1]
                    y_sample = y_sample[np.sort(unique_y)]
                elif nome == "gamma_above" or nome == "gamma_below":
                    x_sample = np.unique(x_sample)
                    y_sample = np.repeat(y_sample[0], len(x_sample))
                else:
                    y_sample = np.unique(y_sample)
                    x_sample = np.repeat(x_sample[0], len(y_sample))


                thickness = 18 * np.ones(len(x_sample)) #dentro ones va la lunghezza del vettore che contiene tutti i punti di allenamento nel caso di grid sono 900 in meno perche la radice di 50 si approssima a 7
                plt.scatter(x_sample, y_sample, s=thickness, color="r", alpha=0.8, marker='+')
        plt.tight_layout()
        plt.savefig(str + '.png', transparent=True, dpi=300)
        plt.close()

    #Questo serve quando devo plottare i punti nel caso in cui uso grid, perché in questo caso ci sono problemi
    def plots_grid(self, val, x, y, str, sample_points):
        xi = np.linspace(0,1,1000)
        yi = np.linspace(0,2,1000)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), val, (xi, yi), method='cubic')
        plt.imshow(zi, extent=(0, 1, 0, 2), origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar()
        #Qua devo plottare tutti i sample points, per farlo uso un ciclo for sui nomi che ho dato ai vari pezzi del dominio
        nomi = ["gamma_above","gamma_below","gamma_right","gamma_left","D"]
        if True: #cioè se voglio mettere anche i punti di sample
            for nome in nomi:
                x_sample = sample_points[nome]['x'].detach().numpy()
                y_sample = sample_points[nome]['y'].detach().numpy()

                #Questa roba serve solo per fare in modo che i grafici vengano più nitidi nei punti di training
                if nome == "D":
                    unique_x = np.unique(x_sample, return_index=True)[1]
                    x_sample = x_sample[np.sort(unique_x)]
                    unique_y = np.unique(y_sample, return_index=True)[1]
                    y_sample = y_sample[np.sort(unique_y)]
                    x_sample, y_sample = np.meshgrid(x_sample, y_sample)
                elif nome == "gamma_above" or nome == "gamma_below":
                    x_sample = np.unique(x_sample)
                    y_sample = np.repeat(y_sample[0], len(x_sample))
                else:
                    y_sample = np.unique(y_sample)
                    x_sample = np.repeat(x_sample[0], len(y_sample))
                thickness = 18 * np.ones(x_sample.shape) #dentro ones va la lunghezza del vettore che contiene tutti i punti di allenamento nel caso di grid sono 900 in meno perche la radice di 50 si approssima a 7
                plt.scatter(x_sample, y_sample, s=thickness, color="r", alpha=0.8, marker='+')
        plt.tight_layout()
        plt.savefig(str + '.png', transparent=True, dpi=300)
        plt.close()
