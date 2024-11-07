import pandas as pd


def df_dict_to_excel(df_dict: dict[str, pd.DataFrame], file_path, sheet_name='results'):
    writer = pd.ExcelWriter(file_path)
    start = 0
    for k, df in df_dict.items():
        df2 = pd.concat({k: df}, axis=1)
        df2.to_excel(writer, sheet_name=sheet_name, startrow=start)
        start += len(df) + df.columns.nlevels + int(bool(df.index.name)) + 3
    writer.save()


def df_min_max(ds) -> tuple[float, float]:
    cols: list[str] = list(ds.columns)
    min_value = None
    max_value = None
    for col in cols:
        col_max = max(ds[col])
        if max_value is None or col_max > max_value:
            max_value = col_max
        col_min = min(ds[col])
        if min_value is None or col_min < min_value:
            min_value = col_min
    return min_value, max_value