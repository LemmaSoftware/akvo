def plotLogo(ax):
    #import matplotlib.pyplot as ax 
    import numpy as np 

    x  = np.arange(-np.pi/2, 7.0*np.pi, .01)
    y  = np.sin(x)
    y2 = np.cos(x)

    y3 = np.sin(.5*x)
    y4 = np.cos(.5*x)

    ax.plot(x,  y, linewidth=1, color='lightgrey')
    ax.plot(x, -y, linewidth=1, color='lightgrey')
    ax.plot(x, y2, linewidth=1, color='lightgrey')
    ax.plot(x,-y2, linewidth=1, color='lightgrey')

    ax.plot(x,  y3, linewidth=1, color='lightgrey')
    ax.plot(x, -y3, linewidth=1, color='lightgrey')
    ax.plot(x,  y4, linewidth=1, color='lightgrey')
    ax.plot(x, -y4, linewidth=1, color='lightgrey')

    a = np.arange(10,615)
    ax.plot(x[a],y[a], linewidth=3, color='black')
    ax.fill_between(x[a],y[a], y2=-1, where=y[a]>=-1, interpolate=True, linewidth=0, alpha=.95, color='maroon')

    k  = np.arange(615, 785)
    k2 = np.arange(785, 1105)
    ax.plot(x[k],    y[k ], linewidth=3, color='black')
    ax.plot(x[k],   -y[k ], linewidth=3, color='black')
    ax.plot(x[k2],  y3[k2], linewidth=3, color='black')
    ax.plot(x[k2], -y3[k2], linewidth=3, color='black')

    #v = np.arange(1265,1865) # y 
    v = np.arange(1105, 1720)
    ax.plot(x[v], -y2[v], linewidth=3, color='black')

    o = np.arange(1728,2357)
    ax.plot(x[o],  y4[o], linewidth=3, color='black')
    ax.plot(x[o], -y4[o], linewidth=3, color='black')
    ax.fill_between(x[o],  y4[o], y2=-y4[o],  where=y4[o]<=1, interpolate=True, linewidth=0, alpha=.95, color='maroon')

if __name__ == "__main__":
    import matplotlib.pyplot as plt 

    fig = plt.figure( figsize=(6,3) )
    #ax = fig.add_axes([.1,.1,.8,.8])
    ax = fig.add_subplot(211)

    fig.patch.set_facecolor( None )
    fig.patch.set_alpha( .0 )
    ax.axis('off')
    plotLogo(ax)
    ax.xaxis.set_major_locator(plt.NullLocator()) 
    ax.yaxis.set_major_locator(plt.NullLocator()) 

    subplot2 = fig.add_subplot(212)
    subplot2.text(0.5, 1.,'surface NMR workbench',
            horizontalalignment='center',
            verticalalignment='center',
            size=22,
            transform = subplot2.transAxes)
    subplot2.xaxis.set_major_locator(plt.NullLocator()) 
    subplot2.yaxis.set_major_locator(plt.NullLocator()) 
    subplot2.axis('off')
    plt.savefig("logo.pdf")
    plt.show()

#ax.fill_between(x[o], -y4[o], y2=0, where=-y4[o]<=1, interpolate=True, linewidth=0, alpha=.5, color='black')
#ax.plot(x[o], y2[o], linewidth=3, color='black')
#ax.plot(x[o],-y2[o], linewidth=3, color='black')
#ax.fill_between(x[a], y[a], y2=-1, where=y[a]>=-1, interpolate=True, linewidth=0, alpha=.5, color='black')

#ax.show()
