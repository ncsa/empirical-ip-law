import pandas as pd
import openpyxl
from bs4 import BeautifulSoup
import docx
import json
import os.path

# make sure dir names is end with one '/'. 
def dir_valid(input_dir):
  if input_dir[-1]!='/':
    input_dir = input_dir + '/'
  return input_dir

# file_unit: submission posts starts with RS_, comments starts with RC_
# ordered by year and month RS[C]_yr-mt
def get_file_unit(yr, mt, header='RS_'):
  if isinstance(mt, int):
    file_unit = header + str(yr) + '-' + str(mt).zfill(2)
  else:
    file_unit = header + str(yr) + '-' + mt
  return file_unit

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
def find_xml_underline(input_dir, file_unit, col_num = 3):
  
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

# .docx, underlined texts in long selftexts
def find_doc_underline(input_dir, id0):
  input_dir = dir_valid(input_dir)
  doc_name = input_dir + id0 + '.docx'

  # if file is not created
  if not os.path.isfile(docx_name):
    return ''

  doc = docx.Document(docx_name)

  underlined_text = []
  for paragraph in doc.paragraphs:
    for run in paragraph.runs:
      if run.font.underline:
        underlined_text.append(run.text)

  return ' '.join(underlined_text) # need to find a better separater

# .json, get long selftexts
# idx_long is obtained from df_body
# idx_long = df_body[df_body['selftext'].str.startswith('30000+')].index.tolist()
# but drop url, title, selftext
# input_preproc_dir = dir_valid(input_preproc_dir)

def pad_json(input_json_dir, input_doc_dir, file_unit, idx_long):
  input_json_dir = dir_valid(input_json_dir)
  input_doc_dir = dir_valid(input_doc_dir)
  file_json = input_json_dir + file_unit + '.json'

  dict_selftext = {}
  dict_underline = {}

  with open(file_json, 'r', encoding = 'utf-8') as f:
    json_data = json.load(f)

    for ii in range(len(json_data)):
      df0 = pd.json_normalize(json_data[ii])
      id0 = df0.at[ii,'id']
      if ii in idx_long:
        dict_selftext = df0.at[ii, 'selftext']
        dict_underline[id0] = find_doc_underline(input_doc_dir, id0)

  return {'selftext_mapping': dict_selftext, 'underline_mapping': dict_underline}

# read the files stored in sub_col directires for cov fields
def read_cov(input_dir, file_unit):
  input_dir = dir_valid(input_dir)
  df = pd.read_sv(input_dir + file_unit + '_sub.csv').drop(columns=['title', 'selftext'])
  return df

# then, put the dfs together
# constants: directories, input_xlsx_dir, input_xml_dir, input_json_dir, input_doc_dir
# iterate yr and mt
def df_annotation_combine(yr, mt, input_xlsx_dir, input_xml_dir, input_json_dir, input_doc_dir, input_cov_dir):
  # generate file_unit
  file_unit = get_file_uniit(yr,mt)

  # load_xlsx
  df_body = load_xlsx_file(input_xlsx_dir, file_unit, ncol_sel = 12, sel_sheet='Sheet1')
  # obtain underlined text from xlm file
  df_underline_short = load_xml_underline(input_xml_dir, file_unit, col_num=3)
  # combine
  df_texts = pd.concat([df_body, df_underline_short], axis=1)

  # obtian index of rows that has long selftexts
  idx_long = df_body[df_body['selftext'].str.startswith('30000+')].index.tolist()
  # go to json file to extract long selftext
  # go to docx file to extract underlines in long selftext
  map_long = pad_json(input_json_dir, input_doc_dir, file_unit, idx_long, col_sel)

  # put long selftexts and its underlines into the dataframe
  for idx, item in df_texts.iterrows():
    id0 = item['id']
      if id0 in idx_long:
        df_texts.at[idx, 'sefltext'] = map_long['dict_selftext'][id0]
        df_texts.at[idx, 'underlined'] = map_long['dict_underline'][id0]

  # read and pad covariates
  df_cov = read_cov(input_cov_dir, file_unit)
  return pd.concat([df_cov, df_texts])


# descriptive stats function to produce figures

# tokenization with LLMs

# train-val-test set splitting - random vs masked output
