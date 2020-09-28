
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plot
import pandas as ps


def main():
    dataset = ps.read_csv('Data.csv')
    x = dataset.iloc[:, :-1].values  # index locate take all cols except the last one indepnedent cols
    y = dataset.iloc[:, -1].values  # get the dependent col last col
    # To fill the empty data slots
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(missing_values=np.nan,strategy="mean")
    imp.fit(x[:, 1:3])
    x[:, 1:3] = imp.transform(x[:, 1:3])

    # encoding independent
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    transformer = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
    x = np.array(transformer.fit_transform(x))

    # encoding dependent
    from sklearn.preprocessing import LabelEncoder
    labelEncoder = LabelEncoder()
    y = labelEncoder.fit_transform(y)

    # splitting data
    from sklearn.model_selection import train_test_split
    train_x, test_x,train_y, test_x =  train_test_split(x, y, test_size= 0.2,random_state= 1)

    # feature scaling
    from sklearn.preprocessing import StandardScaler
    scalar = StandardScaler()
    train_x[:,3:] = scalar.fit_transform(train_x[:,3:])
    test_x[:,3:] = scalar.transform(test_x[:,3:])




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


