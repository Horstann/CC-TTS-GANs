import pandas as pd
import numpy as np
import glob
import os

def main(dir='data/option_chains'):
    for filepath in glob.glob(dir+'/*.txt'):
        os.rename(filepath, filepath[:-4]+'.csv')

    main_df = pd.DataFrame(columns=['maturity', 'moneyness_level', 'underlying_price', 'strike', 'call_price', 'put_price'])

    for filepath in glob.glob(dir+'/*.csv'):
        df = None
        try:
            df = pd.read_csv(filepath, usecols=['Date', 'maturity', 'moneyness_level', 'underlying_price', 'strike', 'call_price', 'put_price', ])
            df['Date'] = pd.to_datetime(df['Date'])
        except ValueError:
            df = pd.read_csv(filepath, usecols=[' [QUOTE_DATE]', ' [EXPIRE_DATE]', ' [STRIKE]', ' [UNDERLYING_LAST]', ' [C_LAST]', ' [P_LAST]'])
            df = df.rename(columns={
                ' [QUOTE_DATE]': 'Date', 
                ' [EXPIRE_DATE]': 'maturity', 
                ' [STRIKE]': 'strike', 
                ' [UNDERLYING_LAST]': 'underlying_price', 
                ' [C_LAST]': 'call_price', 
                ' [P_LAST]': 'put_price'
            })
            df['Date'] = pd.to_datetime(df['Date'])
            df['maturity'] = pd.to_datetime(df['maturity'])
            df['maturity'] = (df['maturity'] - df['Date']) / np.timedelta64(1, 'D')
            df['moneyness_level'] = df['strike'] / df['underlying_price']
            df = df[(df['call_price']!=' ') & (df['put_price']!=' ')]
            df['call_price'] = df['call_price'].astype(float)
            df['put_price'] = df['put_price'].astype(float)
            df = df[(df['maturity']>=1) & ((df['call_price']>0) | (df['put_price']>0))]
            
            df.to_csv(filepath, index=False)
        
        old_dates = set(df['Date'].copy())
        df = df[(df['maturity']>=30) & (df['maturity']<=90)]
        assert not df.empty, f"{filepath} Dataframe found empty after filtering on maturity."
        df = df.groupby(['Date']).mean()
        new_dates = set(df.index.copy())
        assert len(old_dates) == len(old_dates.intersection(new_dates)), f"{filepath} Dataframe lost some dates after filtering on maturity."

        main_df = pd.concat([main_df, df])
    
    main_df.index.name = 'Date'
    
    print(main_df)
    main_df.to_csv('data/option_chains.csv')

if __name__ == "__main__":
    main()