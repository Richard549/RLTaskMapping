import sys
import matplotlib.pyplot as plt
import numpy as np

times1 = []
times2 = []
times3 = []

with open(sys.argv[1],'r') as f:
	for line in f:
		if "Benchmark execution time at episode" in line:
			time_str = line.split(",reward")[0].split(" ")[-1]
			times1.append(float(time_str))

with open(sys.argv[2],'r') as f:
	for line in f:
		if "Benchmark execution time at episode" in line:
			time_str = line.split(",reward")[0].split(" ")[-1]
			times2.append(float(time_str))

with open(sys.argv[3],'r') as f:
	for line in f:
		if "Benchmark execution time at episode" in line:
			time_str = line.split(",reward")[0].split(" ")[-1]
			times3.append(float(time_str))

times1 = np.array(times1)
times2 = np.array(times2)
times3 = np.array(times3)

print(times1.shape)
print(times2.shape)
print(times3.shape)

times = np.mean(np.array([times1, times2]), axis=0)

window = 1
if any(list(("window=" in sys.argv[i]) == True for i in range(len(sys.argv)))):
	for i in range(len(sys.argv)):
		if "window=" in sys.argv[i]:
			window = int(sys.argv[i].split("=")[1])
			break

cumsum, moving_aves = [0], []

for i, x in enumerate(times, 1):
	cumsum.append(cumsum[i-1] + x)
	if i>=window:
		moving_ave = (cumsum[i] - cumsum[i-window])/window
		moving_aves.append(moving_ave)


if "graph" in sys.argv:
	plt.scatter(range(len(moving_aves)),moving_aves,s=2)

if "graph" in sys.argv:

	if any(list(("xlim=" in sys.argv[i]) == True for i in range(len(sys.argv)))):
		for i in range(len(sys.argv)):
			if "xlim=" in sys.argv[i]:
				lower = float(sys.argv[i].split("=")[1].split(",")[0])
				upper = float(sys.argv[i].split("=")[1].split(",")[1])
				plt.xlim([lower,upper])
				break
	if any(list(("ylim=" in sys.argv[i]) == True for i in range(len(sys.argv)))):
		for i in range(len(sys.argv)):
			if "ylim=" in sys.argv[i]:
				lower = float(sys.argv[i].split("=")[1].split(",")[0])
				upper = float(sys.argv[i].split("=")[1].split(",")[1])
				plt.ylim([lower,upper])
				break

	plt.title(sys.argv[4])
	plt.xlabel(sys.argv[5])
	plt.ylabel(sys.argv[6])
		
	plt.show()
