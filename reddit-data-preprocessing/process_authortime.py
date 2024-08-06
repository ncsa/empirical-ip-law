import pandas as pd
import datetime 

def convert_time(time_val):
    dt = datetime.datetime.utcfromtimestamp(time_val)
    readable_time = dt.strftime('%Y-%m-%d %H:%M:%S')
    return readable_time

author_hash = pd.read_csv('author_hash.csv')
author_hash_dict = dict(zip(author_hash['id'],author_hash['uuid']))

in_dir = 'csv_0806/full_csv_rep/'
out_dir = 'csv_0806/converted_csv_rep/'


for yr in ['2022','2023']:
    for mt in ['01','02','03','04',
               '05','06','07','08',
               '09','10','11','12']:
        file_header = 'RS_' + yr + '-' + mt
        fin = in_dir + file_header + '_rep.csv'
        fout = out_dir + file_header + '_conv.csv'
        df = pd.read_csv(fin)
        df['author'] = df['author'].apply(lambda x: author_hash_dict[x])
        df['created_utc'] = df['created_utc'].apply(convert_time)
        df.to_csv(fout,index=False)
