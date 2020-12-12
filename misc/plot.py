import csv

cval = []
c_p = []
c_r = []
reader = csv.reader(open('test_c.txt', 'rb'), delimiter=' ')
for c,p,r in reader:
    cval.append(float(c))
    c_p.append(float(p))
    c_r.append(float(r))

jval = []
j_p = []
j_r = []
reader = csv.reader(open('test_j.txt', 'rb'), delimiter=' ')
for j,p,r in reader:
    jval.append(float(j))
    j_p.append(float(p))
    j_r.append(float(r))

import matplotlib.pyplot as plt

plt.subplot(2,1,1)
plt.plot(cval, c_p, label='precision')
plt.plot(cval, c_r, label='recall')
plt.xlabel('value of parameter `c\'')
plt.legend()

plt.subplot(2,1,2)
plt.plot(jval, j_p, label='precision')
plt.plot(jval, j_r, label='recall')
plt.xlabel('value of parameter `j\'')
plt.legend()

plt.show()
