import copclib as copc
import argparse
import statistics as stats
import matplotlib.pyplot as plt
from os import path

# Argument parser erstellen
parser = argparse.ArgumentParser(description="Ein Script welches eine Statistik über die Nodes einer Copc Datei erstellt")
parser.add_argument("-i", "--input", type=str, nargs="+", help="Pfad zu ein oder mehreren copc Datein: Wenn mehrere Dateien übergeben werden werden keine Histogramme erstellt")
parser.add_argument("-o", "--output", dest="output", default="./statistics_summary.csv", type=str, help="Pfad zur csv File")
parser.add_argument("-s", "--single_file", default=False, type=bool, help="write single file to csv")
args = parser.parse_args()

readers, nodes_list, point_counts_lists, minimums, maximums, mediums, medians, stds = [], [], [], [], [], [], [], []
for i, file in enumerate(args.input):
    readers.append(copc.FileReader(file))
    nodes_list.append(readers[i].GetAllNodes())

    point_counts_lists.append([node.point_count for node in nodes_list[i]])

    minimums.append(min(point_counts_lists[i]))
    maximums.append(max(point_counts_lists[i]))
    mediums.append(stats.mean(point_counts_lists[i]))
    medians.append(stats.median(point_counts_lists[i]))
    stds.append(stats.stdev(point_counts_lists[i]))

if (len(args.input) == 1) and (not args.single_file):
    text = f"""
    Statistik:
    Anzahl der Punkte:  {sum(point_counts_lists[0])}
    Anzahl der Nodes:   {len(nodes_list[0])}
    Maximale Tiefe:     {readers[0].GetMaxDepth()}

    Minimum: {minimums[0]}
    Maximum: {maximums[0]}
    Minimum (ohne 0): {min([x for x in point_counts_lists[0] if x != 0])}

    Durchschnitt:   {round(mediums[0])}
    Median:         {medians[0]}

    Standardabweichung: {round(stds[0],1)}
    """
    print(text)

    plt.figure(figsize=(14, 6))
    plt.hist(point_counts_lists[0], bins=30, color='skyblue', edgecolor='black')

    plt.text(1.02, 0.95, text, 
            transform=plt.gca().transAxes, 
            fontsize=10, 
            verticalalignment='top', 
            horizontalalignment='left',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)) # Weiße Box mit Transparenz

    plt.subplots_adjust(right=0.7) 

    plt.title(f'Histogram der Punktanzahlen\n{args.input[0]}')
    plt.xlabel('Anzahl der Punkte')
    plt.ylabel('Häufigkeit')
    plt.grid()
    plt.show()

if args.single_file or (len(args.input) != 1):
    if not path.exists(args.output):
        with open(args.output, "w") as ofile:
            ofile.write("name,npoints,nnodes,max_depth,min,max,mean,median,std\n")
    
    with open(args.output, "a") as ofile:
        for i in range(len(args.input)):
            ofile.write(f"{args.input[i]},{sum(point_counts_lists[i])},{len(nodes_list[i])},{readers[i].GetMaxDepth()},{minimums[i]},{maximums[i]},{round(mediums[i])},{medians[i]},{round(stds[i], 1)}\n")