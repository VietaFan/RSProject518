import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import json

output = open("output.txt", "r")
data = []

for line in output:
  # never write a line like this ever again Owen
  data.append(json.loads(line.replace("'", '"')))

# divide into groups with same a
alphas = [[]]
exps = [[]]
utilities = [[]]
a_list = []

# keep track of As
first_a = data[0]["cf_a"]
prev_a = first_a
a_list.append(first_a)

for point in data:
  # if we switch As
  if point["cf_a"] != prev_a:
    prev_a = point["cf_a"]
    a_list.append(point["cf_a"])

    alphas.append([])
    exps.append([])
    utilities.append([])

  alphas[-1].append(point["cf_alpha"])
  exps[-1].append(point["weighting_exp"])
  utilities[-1].append(point["utility"])

# make the plots
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(alphas, exps, utilities, c=a_list)
# plt.show()

ax_arr = []
for idx, a in enumerate(a_list):
  fig = plt.figure(idx)
  ax_arr.append(plt.axes(projection='3d'))

  curr_ax = ax_arr[idx]
  curr_ax.scatter3D(alphas[idx], exps[idx], utilities[idx])

  curr_ax.set_xlabel("alpha")
  curr_ax.set_ylabel("weighting exponent")
  curr_ax.set_zlabel("utility")
  curr_ax.set_title("Utilities where a = " + str(a_list[idx]))

plt.show()

