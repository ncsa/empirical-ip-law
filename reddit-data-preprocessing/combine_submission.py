import json
import pandas as pd
import csv
import warnings
warnings.filterwarnings('ignore')

def json_to_dfs(in_dir, yr, mt):
    file_header = 'RS_' + yr + '-' + mt
    fin = in_dir + file_header + '.json' 

    with open(fin, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    first_occurrence_local = {}	
    df_list = []

    for ii in range(len(json_data)):
        df0 = pd.json_normalize(json_data[ii])
        df0['first_local'] = pd.Series(dtype='str')
        df0['first_global'] = pd.Series(dtype='str')

        cols = df0.columns.tolist()
        idx = (cols).index('wls')
        df0 = df0[cols[:idx+1]]

        #key = tuple([df0['title'][0],df0['selftext'][0]])
        key = df0['selftext'][0]
        if len(key)==0:
            key = df0['title'][0]
        elif key.strip()=='[deleted]':
            key = df0['title'][0]

        if key not in first_occurrence_local:
            first_local = df0['id'][0]
            first_occurrence_local[key] = first_local
        else:
            first_local = first_occurrence_local[key]

        df0['first_local'] = first_local

        if key not in first_occurrence_global:
            first_global = yr + '-' + mt + '_' + df0['id'][0]
            first_occurrence_global[key] = first_global
        else:
            first_global = first_occurrence_global[key]

        df0['first_global'] = first_global


        if(len(df0['selftext'][0]))>30000:
            df0['selftext']='30000+, refer ' + df0['url'][0]
        elif(len(df0['selftext'][0]))==0:
            df0['selftext']='no text' 

        # pad title that starts with '='
        if(df0['title'][0].startswith('=')):
            df['title'][0] = ''.join(('T',df['title'][0])) 
            
        if len(df0['all_awardings'][0])>0:
            awarding_str = df0['all_awardings'][0]
            awarding_obj = awarding_str[0]
            coin_sum = 0
            for ll in range(len(awarding_str)):
                coin_sum = coin_sum + int(awarding_str[ll]['coin_price'])*int(awarding_str[ll]['count'])
        else:
            coin_sum = 0
        df0['all_awardings'][0] = coin_sum

        df_list.append(df0)
    df_out = pd.concat(df_list, axis = 0, join = 'inner', ignore_index=True)

    return df_out



in_dir = 'register_csv_submissions/'
out_dir = 'csv_out/full_csv/'

first_occurrence_global = {}
df_rep_list = []
file_out_list = []

for yr in ['2022','2023']:
    for mt in ['01','02','03','04',
               '05','06','07','08',
               '09','10','11','12']:
        
        print(yr + '-' + mt)
        df_yr_mt = json_to_dfs(in_dir, yr, mt)
        dict_count_local = df_yr_mt.groupby('first_local')['id'].count().to_dict() 
        df_yr_mt['count_local'] = df_yr_mt['first_local'].map(dict_count_local) 

        sel_count_local = (df_yr_mt['count_local']>1)

        df_yr_mt['count_local'] = df_yr_mt['count_local'].astype('str')
#        sel_first_local = (df_yr_mt['id']==df_yr_mt['first_local'])
        
#        first_local_idxs = df_yr_mt[sel_count_local & sel_first_local].index
#        df_yr_mt.loc[first_local_idxs, 'first_local'] = df_yr_mt.loc[first_local_idxs, 'first_local'].str.upper()
        no_local_rep_idxs = df_yr_mt[~sel_count_local].index
        df_yr_mt.loc[no_local_rep_idxs, 'first_local'] = ''

        df_rep_list.append(df_yr_mt)

        file_out = out_dir + 'RS_' + yr + '-' + mt + '_rep.csv'
        file_out_list.append(file_out)

df_full_rep = pd.concat(df_rep_list, ignore_index=True)
dict_count_global = df_full_rep.groupby('first_global')['id'].count().to_dict()

for ii in range(len(file_out_list)):
    df = df_rep_list[ii]
    file_name = file_out_list[ii] 

    df['count_global'] = df['first_global'].map(dict_count_global)
    sel_count_global = (df['count_global']>1)
    df['count_global'] = df['count_global'].astype('str')
#    sel_first_global = ((yr + '-' + mt + '_' + df['id'])==df['first_global'])

#    first_global_idxs = df[sel_count_global & sel_first_global].index 
#    df.loc[first_global_idxs, 'first_global'] = df.loc[first_global_idxs, 'first_global'].str.upper()

    no_global_rep_idxs = df[~sel_count_global].index
    df.loc[no_global_rep_idxs, 'first_global'] = ''
        
    df.to_csv(file_name, index=False,
              quoting=csv.QUOTE_ALL)

