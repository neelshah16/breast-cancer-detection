import numpy as np
import pandas as pd
from sklearn import model_selection, neighbors

df = pd.read_csv('breast-cancer-wisconsin.data')

#############################################################################
#
#         id  Clump_Thickness  Uniformity_of_Cell_Size  \
# 0  1000025                5                        1
# 1  1002945                5                        4
# 2  1015425                3                        1
# 3  1016277                6                        8
# 4  1017023                4                        1
#
#    Uniformity_of_Cell_Shape  Marginal_Adhesion  Single_EpithelialCellSize  \
# 0                         1                  1                          2
# 1                         4                  5                          7
# 2                         1                  1                          2
# 3                         8                  1                          3
# 4                         1                  3                          2
#
#   Bare_Nuclei  Bland_Chromatin  Normal_Nucleoli  Mitoses  Class
# 0           1                3                1        1      2
# 1          10                3                2        1      2
# 2           2                3                1        1      2
# 3           4                3                7        1      2
# 4           1                3                1        1      2
#
#############################################################################


df.replace("?", -999999, inplace=True)
df.drop(["id"], 1, inplace=True)

X = np.array(df.drop(["Class"], 1))
y = np.array(df["Class"])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(accuracy)







