import copclib as copc
import argparse
import statistics as stats

# Argument parser erstellen
parser = argparse.ArgumentParser(description="Ein Script welches eine Statistik über die Nodes einer Copc Datei erstellt")
parser.add_argument("input", type=str, help="Pfad zur copc Datei")
args = parser.parse_args()

reader = copc.FileReader(args.input)
nodes = reader.GetAllNodes()

point_counts = [node.point_count for node in nodes]

minimum = min(point_counts)
maximum = max(point_counts)
medium = stats.mean(point_counts)
median = stats.median(point_counts)
std = stats.stdev(point_counts)

print(f"""
Statistik:
Anzahl der Punkte:  {sum(point_counts)}
Anzahl der Nodes:   {len(nodes)}

Minimum (ohne leere): {min([x for x in point_counts if x != 0])}
Minimum: {minimum}
Maximum: {maximum}

Durchschnitt:   {round(medium)}
Median:         {median}

Standardabweichung: {round(std,1)}
""")