import matplotlib.pyplot as pl
import numpy as np

from pathlib import Path
here = Path(__file__).parent.parent.resolve()

colors = [np.asarray([218, 215, 213])*1./255,
          np.asarray([0, 101, 189])*1./255,
          np.asarray([0, 0, 0])*1./255,
          np.asarray([227, 114, 34])*1./255
          ]

def plot_displacement(list, times, str, colors, foldernames):
    if False: #len(list)>2:
        colors = [(len(list)-i)/(len(list))*colors[0] + i/(len(list))*colors[1] for i in range(len(list)+1)]
    for i in range(len(list)):
        #pl.plot(times[i], list[i], linewidth=0.6, label=foldernames[i])
        pl.plot(times[i], list[i], color = colors[i], linewidth=0.6, label=foldernames[i])
    pl.axis([0, 30, -0.1, 0.1])
    pl.legend(loc='lower left')
    pl.xlabel("time")
    pl.ylabel("y-displacement of tip of the flap")
    pl.savefig(str)
    pl.close()


def plot_timestep(times, str, colors, foldernames):
    if False: #len(times) > 1:
        colors = [(len(times) - i) / (len(times)) * colors[0] + i / (len(times)) * colors[1] for i in
                  range(len(times)+1)]
    times_max = times[0][0] # assume that first time step of first simulation is the maximal timestep
    for i in range(len(times)):
        times_diff = [times[i][k+1] - times[i][k] for k in range(len(times[i])-1)]
        times_mid = [0.5*(times[i][k+1] + times[i][k]) for k in range(len(times[i])-1)]
        #pl.plot(times_mid, times_diff, linewidth=0.6, label=foldernames[i])
        pl.plot(times_mid, times_diff, color=colors[i], linewidth=0.6, label=foldernames[i])
        pl.axis([0, 15, 0.0, 0.011])
    pl.legend(loc='lower left')
    pl.xlabel("time")
    pl.ylabel("time-step size")
    pl.savefig(str)
    pl.close()

if __name__ == "__main__":
    foldernames = ["Init"] 
    names=  ["initial geometry"] 

    times_list = []
    displacement_list = []

    for i in foldernames:
        str_ = str(here) + "/Output/Forward/" + i
        times_list.append(np.loadtxt(str_ + "/times.txt"))
        displacement_list.append(np.loadtxt(str_ + "/displacementy.txt"))

    plot_displacement(displacement_list, times_list,
                      str(here)+"/Output/displacement_plot.pdf", colors, names)