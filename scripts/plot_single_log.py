import sys
import matplotlib.pyplot as plt
import numpy as np

xlims = []
ylims = []

if any(list(("xlim=" in sys.argv[i]) == True for i in range(len(sys.argv)))):
	for i in range(len(sys.argv)):
		if "xlim=" in sys.argv[i]:
			lower = float(sys.argv[i].split("=")[1].split(",")[0])
			upper = float(sys.argv[i].split("=")[1].split(",")[1])
			xlims.append(lower)
			xlims.append(upper)
			break
if any(list(("ylim=" in sys.argv[i]) == True for i in range(len(sys.argv)))):
	for i in range(len(sys.argv)):
		if "ylim=" in sys.argv[i]:
			lower = float(sys.argv[i].split("=")[1].split(",")[0])
			upper = float(sys.argv[i].split("=")[1].split(",")[1])
			ylims.append(lower)
			ylims.append(upper)
			break

rewards = []
times = []

with open(sys.argv[1],'r') as f:
	for line in f:
		if "Benchmark execution time at episode" in line:
			time_str = line.split(",reward")[0].split(" ")[-1]
			reward_str = line.split(",reward was ")[1].strip()
			#print(line + " GIVES " + str(time_str))
			times.append(float(time_str))
			rewards.append(float(reward_str))
			if len(xlims) > 0 and len(times) > (xlims[1]+500):
				break

window = 1
if any(list(("window=" in sys.argv[i]) == True for i in range(len(sys.argv)))):
	for i in range(len(sys.argv)):
		if "window=" in sys.argv[i]:
			window = int(sys.argv[i].split("=")[1])
			break

cumsum, moving_aves = [0], []

target = times
ylabel = "Execution time " + r'($s^{-1}$)'
if "reward" in sys.argv:
	ylabel = "Reward"
	target = rewards

for i, x in enumerate(target, 1):
	cumsum.append(cumsum[i-1] + x)
	if i>=window:
		moving_ave = (cumsum[i] - cumsum[i-window])/window
		moving_aves.append(moving_ave)


if "graph" in sys.argv:
	plt.scatter(range(len(moving_aves)),moving_aves,s=2)

if "graph" in sys.argv:

	if len(xlims) > 0:
		plt.xlim(xlims)
	if len(ylims) > 0:
		plt.ylim(ylims)

	plt.title(sys.argv[2])
	plt.xlabel(sys.argv[3])
	plt.ylabel(ylabel)
		
	#plt.show()
	#'''
	gcf = plt.gcf()
	default_size = gcf.get_size_inches()
	#gcf.set_size_inches( (default_size[0]*1.0, default_size[1]*2.0) )
	gcf.set_size_inches( (default_size[0]*1.2, default_size[1]*1.0) )
	gcf.savefig(sys.argv[5], format='png', dpi=400, bbox_inches='tight')
	#'''
