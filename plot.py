import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import json

output = open("output.txt", "r")
data = []

for line in output:
  # never write a line like this ever again Owen
  data.append(json.loads(line.replace("'", '"')))

# take the first group with same a
a_list = []
alphas = []
exps = []
utilities = []
for point in data:
  a_list.append(point["cf_a"])
  alphas.append(point["cf_alpha"])
  exps.append(point["weighting_exp"])
  utilities.append(point["utility"])

# plot it
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(alphas, exps, utilities, c=a_list√)
plt.show()
