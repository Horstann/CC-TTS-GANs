import pandas as pd
import re

def main(filename='ivol_surface'):
    df = pd.read_excel(f'data/BBG_templates/{filename}.xlsx')
    new_header = df.iloc[0]
    df = df.iloc[1:,:]
    df.columns = new_header
    df = df.reset_index().drop(columns=['index'])
    assert df.shape[1]%2==0, "Dataframe has odd number of columns"

    series_list = []
    for i in range(0,df.shape[1], 2):
        # Get new column name
        col_name_matches = re.findall(r'PCT_MONEYNESS=(\d+(?:\.\d+)?),EXPIRY=(\d+)', df.columns[i+1])
        assert len(col_name_matches)==1, f"Column name \"{df.columns[i+1]}\" cannot be parsed"
        col_name = col_name_matches[0]
        col_name = col_name[0]+"%" + col_name[1]+"d"
        # Extract series
        series = pd.Series(df.iloc[:,i+1].values, index=pd.to_datetime(df.iloc[:,i], dayfirst=True).values, name=col_name)
        series_list.append(series)
    clean_df = pd.concat(series_list, axis=1)
    clean_df.index.name = "Date"

    print(clean_df)
    print("Number of NaNs:")
    print(clean_df.isna().sum())

    clean_df.to_csv(f'data/{filename}.csv')

if __name__ == "__main__":
    main()