import pandas as pd
col_sel = pd.read_csv('col_sel.txt')['COLUMNS'].tolist()

in_dir = 'csv_out/converted_csv/'
out_dir = 'csv_out/subset/'

for yr in ['2022','2023']:
    for mt in ['01','02','03','04',
               '05','06','07','08',
               '09','10','11','12']:
        file_header = 'RS_' + yr + '-' + mt
        fin = in_dir + file_header + '_conv.csv'
        fout = out_dir + file_header + '_sub.csv'
        df = pd.read_csv(fin)
        df[col_sel].to_csv(fout, index=False)
