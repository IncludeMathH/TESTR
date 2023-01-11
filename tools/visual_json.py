import json
 
filename = "/data1/dn/TESTR/output/TESTR/totaltext/TESTR_R-50_Polygon_attention-loss/metrics.json"
file = open(filename, "rb")
tmp = []
for line in open(filename,'r'):
    tmp.append(json.loads(line))
total_losses = lrs = iters = []
det_F = e2e_F = []
for item in tmp[:5]:
    print(item['lr'], type(item['lr']))
    total_losses.append(item['total_loss'])
    lr = item['lr']
    lrs.append(lr)
    iters.append(item['iteration'])
    if 'DETECTION_ONLY_RESULTS/hmean' in item:
        det_F.append(item["DETECTION_ONLY_RESULTS/hmean"])
        e2e_F.append(item["E2E_RESULTS/hmean"])
print(lrs[:5])
