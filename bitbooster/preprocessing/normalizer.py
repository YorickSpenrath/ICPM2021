import pandas as pd


def normalize(original_data):
    from sklearn.preprocessing import MinMaxScaler
    mms = MinMaxScaler()
    if isinstance(original_data, pd.DataFrame):
        new_data = original_data.to_numpy()
    else:
        new_data = original_data
    new_data = mms.fit_transform(new_data)
    if isinstance(original_data, pd.DataFrame):
        new_data = pd.DataFrame(data=new_data, columns=original_data.columns, index=original_data.index)
