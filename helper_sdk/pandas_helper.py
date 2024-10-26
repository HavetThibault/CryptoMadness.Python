import pandas as pd


def df_dict_to_excel(df_dict: dict[str, pd.DataFrame], file_path, sheet_name='results'):
    writer = pd.ExcelWriter(file_path)
    start = 0
    for k, df in df_dict.items():
        df2 = pd.concat({k: df}, axis=1)
        df2.to_excel(writer, sheet_name=sheet_name, startrow=start)
        start += len(df) + df.columns.nlevels + int(bool(df.index.name)) + 3
    writer.save()
