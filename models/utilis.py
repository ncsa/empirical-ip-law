import pandas as pd
import openpyxl
from bs4 import BeautifulSoup
import json
import os.path

# make sure dir names is end with one '/'. 
def dir_valid(input_dir):
  if input_dir[-1]!='/':
    input_dir = input_dir + '/'
  return input_dir

# workbook should be converted to .xlsx and xml formats
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

# .xml, get underlined texts in selftext
def extract_font(input_dir, file_unit, col_num = 3):
  
  input_dir = dir_valid(input_dir)

  file = open(input_dir + file_unit + '.xml', 'r')

  # parsing
  soup = BeautifulSoup(contents, 'xml')
 
  # iterate over rows
  rows = soup.find_all('Row')

  underline_list = []
  for row in rows:
    # selftext is on column D, therefore col_num = 4-1 = 3
    cells = row.find_all('Cell')
    if len(cells)<4:
      sel_texts = ''
    else:
      cell_selftext = cells[col_num]
      # identify texts enclosed in <Font><U> ... </Font></U>
      underlined = cell_selftext.find_all('U')
      sel_texts = ' '.join([x.text for x in underlined]) # need to find a better separater
    underlined_list.append(sel_texts)
    df = pd.DataFrame()
    df['underlined'] = pd.Series(underlined_list)
    return df
  

# .json, get long selftexts

# .docx, underlined texts in long selftexts

# descriptive stats function to produce figures

# tokenization with LLMs

# train-val-test set splitting - random vs masked output
