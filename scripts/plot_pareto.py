import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

N = 64

# Cost vs runtime for each alpha given deadline
p = 0.6

plt.figure(figsize=(5, 4))
plt.xlim(1, 2)
plt.ylim(0.3, 1.05)
plt.xlabel('expected relative runtime $\\mathbb{E}[runtime]/\\theta_0$')
plt.ylabel('expected savings (%)')

# theta = [1.25, 1.2, 1.15, 1.1, 1.05]
theta = [1.1]
colors = ['blue', 'teal', 'green']
for t in range(len(theta)):
    a = np.linspace(0.5, 1, 10, endpoint=False)


    plt.vlines(theta[t], ymin=0, ymax=2, color=colors[t], alpha=0.3)#, label="$\\theta={:.2f}$".format(theta[t]))
    # time = []
    # cost = []
    nstar = np.minimum(np.floor(np.reciprocal(1-a)*N*(1-1/theta[t])), N)
    runtime = np.reciprocal(N - np.outer(1-a, nstar)) * N
    cost = np.divide(np.outer(a*p - 1, nstar) + N, np.multiply(a - 1, nstar) + N)
    for i in range(a.shape[0]):
        plt.plot(runtime[:, i], cost[:, i], color=colors[t], alpha=(a[i]-0.45)*2, marker='|')
        plt.plot(runtime[i, i], cost[i, i], color=colors[t], alpha=(a[i]-0.45)*2, marker='.', label="$\\alpha={:.2f}$".format(a[i]))

plt.legend()
# plt.show()
plt.savefig('../2026_misestimation.pdf')


# Cost vs N_s for each alpha
p = 0.5

plt.figure(figsize=(4.5, 3))
plt.xlim(0, N)
plt.ylim(0.4, 1)
plt.xlabel('spot instances $N_s$')
plt.ylabel('cost savings (\%)')

ns = np.arange(N+1)
a = np.linspace(0.5, 1, 5, endpoint=False)
num = np.outer((a*p) - 1, ns) + N
den = np.outer(a-1, ns) + N
e_cost = np.divide(num, den)
nstar = np.argmin(e_cost, axis=1)

for i in range(a.shape[0]):
    # print(e_cost[n, :], ns)
    plt.plot(ns, e_cost[i, :], color='black', alpha=(a[i]-0.4)*2, label="$\\alpha={}$".format(a[i]))
    plt.plot(nstar[i], e_cost[i, nstar[i]], color='black', alpha=(a[i]-0.4)*2, marker='.')

plt.legend()
# plt.show()