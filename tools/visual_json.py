import json
import copy
import matplotlib.pyplot as plt

dir = '/work/dn/TESTR/output/TESTR/totaltext/Bezier-Loss-in-True-field+mask-guided-offsets'
filename = dir + "/metrics.json"

file = open(filename, "rb")
# 读取每一行
tmp = []
for line in open(filename, 'r'):
    tmp.append(json.loads(line))

total_losses = []
lrs = []
iters = []
det_F = []
e2e_F = []
for t in tmp:
    total_losses.append(t['total_loss'])
    lrs.append(t['lr'])
    iters.append(t['iteration'])
    if 'DETECTION_ONLY_RESULTS/hmean' in t:
        det_F.append(t['DETECTION_ONLY_RESULTS/hmean'])
        e2e_F.append(t['E2E_RESULTS/hmean'])
print(f'length of loss:{len(total_losses)}, length of F score:{len(det_F)}, the max of det-F: {max(det_F)}, the max '
      f'of e2e_F: {max(e2e_F)}')

fig1 = plt.figure('figure1')
# plt.plot(list(range(1, len(total_losses)+1)), total_losses, color = 'orange', label = 'total_losses')
plt.plot(iters, total_losses, color='orange', label='total_losses')
plt.xlabel("number of iters")
plt.ylabel("loss")
plt.legend(loc=1)
plt.show()
fig1.savefig(dir + '/figure-loss.png')

fig2 = plt.figure('figure2')
plt.plot(list(range(1, len(det_F)+1)), det_F, color = 'orange', label = 'det hmean')
plt.plot(list(range(1, len(e2e_F)+1)), e2e_F, color = 'blue', label = 'e2e hmean')
plt.xlabel("number of iters")
plt.ylabel("hmean")
plt.legend()
plt.show()
fig2.savefig(dir+'/figure-hmean.png')

fig3 = plt.figure('figure3')
# plt.plot(list(range(1, len(total_losses)+1)), total_losses, color = 'orange', label = 'total_losses')
plt.plot(iters, lrs, color = 'orange', label = 'learning rate')
plt.xlabel("number of iters")
plt.ylabel("learning rate")
plt.legend()
plt.show()
fig3.savefig(dir + '/figure-learning_rate.png')
