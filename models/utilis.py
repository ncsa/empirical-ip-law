import pandas as pd
import openpyxl
import json
import os.path

# make sure dir names is end with one '/'. 
def dir_valid(input_dir):
  if input_dir[-1]!='/':
    input_dir = input_dir + '/'
  return input_dir

# workbook should be converted to .xlsx and xlm formats
# also include .json files to paste back long selftexts
# additionally, there will be .docx files including underlined texts in long selftext

# .xlsx, get manual annotations
# ncol_sel defaulted to 12 columns, set to 16 when duplicate text count cols are included
def load_xlsx_file(input_dir, file_unit, ncol_sel = 12, sel_sheet='Sheet1'):
  input_dir = dir_valid(input_dir)
  wb = openpyxl.load_workbook(input_dir + file_unit + '.xlsx')
  ws = wb[sel_sheet]

  # convert to dataframe
  df = pd.DataFrame(ws.values).loc[:,0:ncol_sel]
  df_header = df.iloc[0]

  # remove tailing space in col names
  df_header = [x.strip() for x in df_header]
  df_body = df[1:]
  df_body.columns = df_header

  # fill NA with 'n/a'
  df_body = df_body.fillna('n/a')
  # replace "#" with 'n/a'
  df_body = df_body.mask((df_body=='#'), 'n/a')

  return df_body

# .xlm, get underlined texts in selftext

# .json, get long selftexts

# .docx, underlined texts in long selftexts

# descriptive stats function to produce figures

# tokenization with LLMs

# train-val-test set splitting - random vs masked output
