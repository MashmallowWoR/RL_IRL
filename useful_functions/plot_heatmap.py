def plot_heatmap(data, title):
    heatmap = plt.pcolor(data, cmap='plasma')
    plt.colorbar()
    plt.gca().invert_yaxis() #plt.gca(): get the current polar axes on the current figure
    plt.title(title)
    plt.show()