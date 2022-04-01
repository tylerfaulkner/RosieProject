from os import listdir
import re
import numpy as np
import pandas as pd

train_path = "/data/datasets/affectNet/train_set/"
path = "/data/datasets/affectNet/train_set/annotations/"
img_path = "/data/datasets/affectNet/train_set/images/"
files = listdir(path)
exp = np.array(list(filter(lambda c: 'exp' in c, files)))
# print(list(exp))
nums = re.compile('\d*')

y_train = np.empty([len(exp), 2], dtype=np.int32)

df = pd.DataFrame(columns=['path', 'label'])

for i in range((exp.shape[0])):
    file = exp[i]
    num = nums.match(file)[0]
    array = np.load(path + file)
    # print(array)
    df = df.append({'path': img_path + str(num) + ".jpg", "label": array}, ignore_index=True)
    if i%5000==0:
        print(df.shape)

print("Saving...")
df.to_csv(train_path+"data.csv", index=False)
print("Saved.")

