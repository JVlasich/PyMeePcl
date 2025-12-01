import copclib as copc
import argparse
import statistics as stats
import matplotlib.pyplot as plt
from os import path
import os
import glob

# Argument parser erstellen
parser = argparse.ArgumentParser(description="Ein Script welches eine Statistik über die Nodes einer Copc Datei erstellt")
parser.add_argument("-i", "--input", type=str, nargs="+", help="Pfad zu ein oder mehreren copc Datein: Wenn mehrere Dateien übergeben werden werden keine Histogramme erstellt")
parser.add_argument("-o", "--output", default="./statistics_summary.csv", type=str, help="Pfad zur csv File")
parser.add_argument("-wh", "--write_histogram", default=None, type=str, help="Speichert alle erstellten histogramme wenn ein Verzeichnis angegeben wird")
parser.add_argument("-sh", "--show_histogram", action="store_true", help="Gibt ein Histogram am Bildschirm aus")
args = parser.parse_args()

# check ob input ein folder ist, funktioniert noch nicht wenn ordner nicht ./ ist
if len(args.input) == 1:
    if os.path.isdir(args.input[0]):
        files = glob.glob("*.copc.laz", root_dir=args.input[0])
        args.input = files

readers, nodes_list, point_counts_lists, minimums, maximums, mediums, medians, stds = [], [], [], [], [], [], [], []
for i, file in enumerate(args.input):
    readers.append(copc.FileReader(file))
    nodes_list.append(readers[i].GetAllNodes())

    point_counts_lists.append([node.point_count for node in nodes_list[i] if node.point_count != 0])

    minimums.append(min(point_counts_lists[i]))
    maximums.append(max(point_counts_lists[i]))
    mediums.append(stats.mean(point_counts_lists[i]))
    medians.append(stats.median(point_counts_lists[i]))
    stds.append(stats.stdev(point_counts_lists[i]))

if (len(args.input) == 1) or (args.write_histogram):

    for i in range(len(args.input)):
        text = f"""
        Statistik:
        Anzahl der Punkte:  {sum(point_counts_lists[i])}
        Anzahl der Nodes:   {len(nodes_list[i])}
        davon leere Nodes:  {len(nodes_list[i]) - len([x for x in point_counts_lists[i] if x != 0])}

        Maximale Tiefe:     {readers[i].GetMaxDepth()}

        Minimum: {minimums[i]}
        Maximum: {maximums[i]}

        Durchschnitt:   {round(mediums[i])}
        Median:         {medians[i]}

        Standardabweichung: {round(stds[i],1)}
        """
        #print(text)

        plt.figure(figsize=(14, 6))
        plt.hist(point_counts_lists[i], bins=30, color='skyblue', edgecolor='black')

        plt.text(1.02, 0.95, text, 
                transform=plt.gca().transAxes, 
                fontsize=10, 
                verticalalignment='top', 
                horizontalalignment='left',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)) # Weiße Box mit Transparenz

        plt.subplots_adjust(right=0.7) 

        plt.title(f'Histogram der Punktanzahlen\n{args.input[i]}')
        plt.xlabel('Anzahl der Punkte')
        plt.ylabel('Häufigkeit')
        plt.grid()

        if args.write_histogram:
            if not path.exists(args.write_histogram):
                os.mkdir(args.write_histogram)
            plt.savefig(f"{args.write_histogram}histogram_{args.input[i][2:]}.png")
        if args.show_histogram:
            plt.show()

if args.output:
    if not path.exists(args.output):
        with open(args.output, "w") as ofile:
            ofile.write("name,npoints,nnodes,nnodes_empty,max_depth,min,max,mean,median,std\n")
    
    with open(args.output, "a") as ofile:
        for i in range(len(args.input)):
            ofile.write(f"{args.input[i]},{sum(point_counts_lists[i])},{len(nodes_list[i])},{len(nodes_list[i]) - len([x for x in point_counts_lists[i] if x != 0])},{readers[i].GetMaxDepth()},{minimums[i]},{maximums[i]},{round(mediums[i])},{medians[i]},{round(stds[i], 1)}\n")