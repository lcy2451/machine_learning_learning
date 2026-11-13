from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

if __name__ == "__main__":
    wine_quality = fetch_ucirepo(id=186)

    # data (as pandas dataframes)
    X = wine_quality.data.features
    y = wine_quality.data.targets

    # 划分训练集和测试集
    x_train, x_test = train_test_split(X, train_size=0.8, test_size=0.2, random_state=42, shuffle=True)
    y_train, y_test = train_test_split(y, train_size=0.8, test_size=0.2, random_state=42, shuffle=True)

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=42, shuffle=True)

    # 划分训练集和验证集以及测试集
    x_train, x_temp, y_train, y_temp = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42, shuffle=True)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, train_size=0.5, random_state=42, shuffle=True)