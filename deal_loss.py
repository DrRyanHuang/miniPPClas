import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
from matplotlib import font_manager

font = "Times New Roman"
plt.rcParams["font.sans-serif"] = font


train_log = "train.log"
with open(train_log) as f:
    log_item = f.read().strip().split("\n")


import re
def starts_ends_with(s_, e_, raw_str):
    res = re.findall("{}[\S\s]*?{}".format(s_, e_), raw_str)
    if len(res) in [1]:
        # 只有一个则返回字符串
        return res[0].replace(s_, "").replace(e_, "")
    
    # 有多个则返回字符串 list
    return [r.replace(s_, "").replace(e_, "") for r in res]


top1_acc_list = []
loss_list = []
for item in log_item:
    # print(item)
    if "ppcls INFO: [Train][Epoch" in item:
        if "][Avg]top1:" in item:
            print(item)
            top1_acc = float(starts_ends_with("]top1: ", ", CEL", item))
            loss = float(starts_ends_with("CELoss: ", ", loss:", item))

            top1_acc_list.append(top1_acc)
            loss_list.append(loss)

plt.plot(top1_acc_list, label="top1_acc")
plt.plot(loss_list, label="loss")
plt.legend()
plt.show()

# for font in font_manager.fontManager.ttflist:
#     # 查看字体名以及对应的字体文件名
#     print(font.name, '-', font.fname)