def plot_table(data):
    fig = plt.figure()
    ax = fig.add_subplot(111)   
    col_labels = list(range(0,10))
    row_labels = [' 0 ', ' 1 ', ' 2 ', ' 3 ', ' 4 ', ' 5 ', ' 6 ', ' 7 ', ' 8 ', ' 9 ']

    # Draw table
    value_table = plt.table(cellText=data, colWidths=[0.05] * 10,
                          rowLabels=row_labels, colLabels=col_labels,
                          loc='center')
    value_table.auto_set_font_size(True)
    value_table.set_fontsize(24)
    value_table.scale(2.5, 2.5)

    # Removing ticks and spines enables you to get the figure only with table
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right','top','bottom','left']:
        plt.gca().spines[pos].set_visible(False)