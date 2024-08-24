import pandas as pd
from bs4 import BeautifulSoup
import docx
import json
import os.path

import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

from transformers import TrainingArguments
from transformers import Trainer

from sklearn import train_test_split

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
  df = pd.read_excel(input_dir + file_unit + '.xlsx', 
    sheet_name = sel_sheet, 
    header=0,
    usecols=range(n:colsel))

  # change header to lower case and remove tailing spaces
  df_header = df.columns.to_list()
  df_new_header = [x.strip().lower() for x in df_header]
  df.rename(columns=dict(zip(df_header, df_new_header)), inplace=True)

  # fill NA with 'n/a'
  df = df.fillna('n/a')
  # replace "#" with 'n/a' 
  # "#" is a temporary option, where annotators are not sure what to put in
  # "#" won't exist in the final annotations.
  df = df.mask((df=='#'), 'n/a')

  return df

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
# some requires combined all yr and mt

# model tokenization
# separate X and y
# title + selftext -> prediction
# title + selftext -> highlight, notes -> prediction
# tokenize with weight when combination is needed
def tokenize_and_weight(text, tokenizer, weight, max_length=128):
  encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
  input_ids = encodings['input_ids']
  attention_mask = encodings['attention_mask']
    
  # Apply weight to tokenized representations
  weighted_input_ids = input_ids * weight
  return {'input_ids': weighted_input_ids, 'attention_mask': attention_mask}

def combine_tokens(tokenizer, df, col_list, weight_list, max_length=128):
  tok_input_id_list = []
  tok_attention_mask_list = []
  for ii in range(len(weight_list)):
    tokenized_X = tokenize_and_weight(df[col_list[ii]].tolist(), tokenizer, weight_list[ii])
    tok_input_id_list.append(tokenized_X['input_ids'])
    tok_attention_mask_list.append(tokenized_X['attention_mask'])

  input_ids_combined = torch.cat(tok_input_id_list, dim=1)
  attention_mask_combined = torch.cat(tok_attention_mask_list, dim=1)
  return {'input_ids': input_ids_combined, 'attention_mask': attention_mask_combined}

# tokenize X and y
#X1 = combine_tokens(tokenizer, df, ['title', 'selftext'], [1.0,1.0])
#X2 = combine_tokens(tokenizer, df, ['underlined', 'notes', 'background'], [1.0, 1.0, 1.0])
#X3 = combined_tokens(tokenizer, df, ['jurisdiction', "poster's legal status", 'category'], [1.0, 1.0])
#y = combined_tokens(tokenizer, df, ['misconceptions', 'unclear knowledge'], [1.0, 1.0])
# X2 and X3 can be used as both X and y
# we can do [X1, X2/3/both] -> y
# or X1 -> X2/3/both -> y
# or combined models

# helper function to store paths to train_val_test splitted tokenized data
def gen_dict_data_path(data_dir, split_type, file_header):
  data_dir = dir_valid(data_vir)
  for key in ['train','val','test']:
    dict_data_path[key] = data_dir + split_type + '_' + file_header + '_' + key + '_data.pth'
  return dict_data_path

# train-val-test set splitting - random vs masked output
# random split
# dict_data_path = gen_dict_data_path(data_dir, 'rnd', file_header)
def random_split(X, y, dict_data_path):
  X_input_ids = X['input_ids']
  X_attention_mask = X['attention_mask']
  y_input_ids = y['input_ids']
  y_attention_mask = y['attention_mask']
  X_train_input_ids, X_test_input_ids, X_train_attention_mask, X_test_attention_mask, y_train_input_ids, y_test_input_ids, y_train_attention_mask, y_test_attention_mask = train_test_split(
    X_input_ids, 
    X_attention_mask,
    y_input_ids,
    y_attention_mask,
    test_size=0.1,  # 10% test data
    random_state=42
  )
  X_train_input_ids, X_val_input_ids, X_train_attention_mask, X_val_attention_mask, y_train_input_ids, y_val_input_ids, y_train_attention_mask, y_val_attention_mask = train_test_split(
      X_train_input_ids,
      X_train_attention_mask,
      y_train_input_ids,
      y_train_attention_mask,
      test_size=0.1,  # 10% validation data
      random_state=42
  )

  torch.save({
      'X_train_input_ids': X_train_input_ids,
      'X_train_attention_mask': X_train_attention_mask,
      'y_train_input_ids': y_train_input_ids,
      'y_train_attention_mask': y_train_attention_mask
  }, dict_data_path['train'])

  torch.save({
      'X_val_input_ids': X_val_input_ids,
      'X_val_attention_mask': X_val_attention_mask,
      'y_val_input_ids': y_val_input_ids,
      'y_val_attention_mask': y_val_attention_mask
  }, dict_data_path['val'])

  torch.save({
      'X_test_input_ids': X_test_input_ids,
      'X_test_attention_mask': X_test_attention_mask,
      'y_test_input_ids': y_test_input_ids,
      'y_test_attention_mask': y_test_attention_mask
  }, dict_data_path['test'])


# masked output - keep one annotation as the unknown test set to the model, see if GenAI can find it
# specific_col: select from "misconception" and "vague knowledge". currently don't do "correct" and "irrelevant"
# currently, do not consider the case if the cols "misconception" and "vague knowledge" are not mutually exclusive
# probably even if they are not mutually exclusive, we can still do them.
# or we can just verify by training on single cols of y again
#  dict_data_path = gen_dict_data_path(data_dir, 'msk', file_header)

def mask_split(X_input_ids, X_attention_mask, 
  y_input_ids, y_attention_mask, 
  df, specific_col, specific_value, dict_data_path):

  y_list = df[specific_col].tolist()
  test_indices = [i for i, label in enumerate(y_list) if label==specific_value]
  train_val_indices = [i for i, label in enumerate(y_list) if label != specific_value]

  X_input_ids_list = X_input_ids.tolist()
  X_attention_mask_list = X_attention_mask.tolist()

  # extract test data
  X_test_input_ids = torch.tensor([input_ids_list[i] for i in test_indices])
  X_test_attention_mask = torch.tensor([attention_mask_list[i] for i in test_indices])
  y_test_input_ids = torch.tensor([y_input_ids[i] for i in test_indices])
  y_test_attention_mask = torch.tensor([y_attention_mask[i] for i in test_indices])

  # extract train-validation data
  X_train_val_input_ids = torch.tensor([input_ids_list[i] for i in train_val_indices])
  X_train_val_attention_mask = torch.tensor([attention_mask_list[i] for i in train_val_indices])
  y_train_val_input_ids = torch.tensor([y_input_ids[i] for i in train_val_indices])
  y_train_val_attention_mask = torch.tensor([y_attention_mask[i] for i in train_val_indices])

  # split train validation randomly
  X_train_input_ids, X_val_input_ids, X_train_attention_mask, X_val_attention_mask, y_train_input_ids, y_val_input_ids, y_train_attention_mask, y_val_attention_mask = train_test_split(
    X_train_val_input_ids,
    X_train_val_attention_mask,
    y_train_val_input_ids,
    y_train_val_attention_mask,
    test_size=0.1,  # 10% validation data
    random_state=42
  )

  torch.save({
      'X_train_input_ids': X_train_input_ids,
      'X_train_attention_mask': X_train_attention_mask,
      'y_train_input_ids': y_train_input_ids,
      'y_train_attention_mask': y_train_attention_mask
  }, dict_data_path['train'])

  torch.save({
      'X_val_input_ids': X_val_input_ids,
      'X_val_attention_mask': X_val_attention_mask,
      'y_val_input_ids': y_val_input_ids,
      'y_val_attention_mask': y_val_attention_mask
  }, dict_data_path['test'])

  torch.save({
      'X_test_input_ids': X_test_input_ids,
      'X_test_attention_mask': X_test_attention_mask,
      'y_test_input_ids': y_test_input_ids,
      'y_test_attention_mask': y_test_attention_mask
  }, dict_data_path['val'])



# train/val/test, rnd_..._data.pth, msk_..._data.pth
# split_type = rnd or msk
# dict_data_path = gen_dict_data_path(data_dir, split_type, file_header)
def run_model_data(model_name, dict_data_path):
  # Load tokenizer and model
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(model_name)

  train_data_path = dict_data_path['train']
  val_data_path = dict_data_path['val']
  test_dataa_path = dict_data_path['test']

  # load data
  train_data = torch.load(train_data_path)
  val_data = torch.load(val_data_path)
  test_data = torch.load(test_data_path)

  # extract data
  X_train_input_ids = train_data['X_train_input_ids']
  X_train_attention_mask = train_data['X_train_attention_mask']
  y_train_input_ids = train_data['y_train_input_ids']
  y_train_attention_mask = train_data['y_train_attention_mask']

  X_val_input_ids = train_data['X_val_input_ids']
  X_val_attention_mask = train_data['X_val_attention_mask']
  y_val_input_ids = train_data['y_val_input_ids']
  y_val_attention_mask = train_data['y_val_attention_mask']

  X_test_input_ids = train_data['X_test_input_ids']
  X_test_attention_mask = train_data['X_test_attention_mask']
  y_test_input_ids = train_data['y_test_input_ids']
  y_test_attention_mask = train_data['y_test_attention_mask']

  # read training args from file_arg_path
  # read data
  trainer = Trainer(
    model = model_name, 
    args = training_args,
    training_dataset = train_dataset,
    eval_dataset = val_dataset
  )
  trainer.train()
  results = trainer.evaluate()
  ## save results
  preds = trainer.predict(X_test_dataset)
  ## save preds
  ## save model
  model.save_pretrained(out_dir_path + 'model')
  tokenizer.save_pretrained(out_dir_path + 'model')

# prompting
# create prompt
# dict_split: indices of train, val, test - random splitting or masked splitting
# in create_prompt, use train set
def create_prompt(title, selftext, df):
  prompt = "Here are some examples:\n"
  for idx, row in df.iterrows():
    ex_title = row['title']
    ex_selftext = row['selftext']
    ex_relevance = row['Relevant']

    prompt += f"Title: {ex_title}\nSelftext: {ex_selftext}\n"

    ex_underline = row['underline']
    ex_note = row['note']
    # what does the background column mean? ask Xiaoren
    ex_bg = row['Background'] 
    # shall i include jurisdiction col?
    ex_ju = row['Jursidictions']

    if len(ex_underline)>0:
      prompt += f"We focused in the part in selftext: {ex_underline}\n"
    if len(ex_note)>0:
      prompt += f"We noted from selftext: {ex_note}\n"
    if len(ex_bg)>0:
      prompt += f"Background is {ex_bg}:\n"
    if len(ex_ju)>0:
      prompt += f"According to jurisdiction in {ex_ju}\n"
    
    if ex_relevance.contains('irrelevant'):
      ex_conclusion='irrelevant'
    else:
      ex_misconception = row['misconception']
      ex_vagueknowledge = row['unclear knowledge']
    ex_conclusion = f"misconception: {ex_misconception}; unclear knowledge: {ex_vagueknowledge}\n\n"
    prompt += "Conclusion:"

  prompt += f"With Title: {title}\n and Selftext: {selftext}, what do we know?"
  prompt += "order results in focused_part, note, conclusion, background"
  # or prompt to show focused_part and note first
  # create two different versions of last prompting: with title and selftext:
  # 1. what is the conclusion?
  # 2. what part in selftext is focused part? what note can we taken? what is the background?
  #    then, what is the conclusion?
  # add the end: show results in tabulated format
  # or, another way is to make focused part, note, and background as result
  # then, write another prompt function, from focus, note, bg, derive conclusion
  return prompt

def generate_conclusion(title, text, model, tokenizer, examples):
  # create the prompt
  # tokenize the prompt -> can i save the example prompts and then add the last line in the prompt?
  inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)

  # Generate the output
  with torch.no_grad():
    outputs = model.generate(
      inputs['input_ids'],
      attention_mask=inputs['attention_mask'],
      max_length=200,  # Adjust based on expected output length
      num_beams=5,     # Number of beams for beam search
      early_stopping=True
  )

  # Decode and return the result
  conclusion = tokenizer.decode(outputs[0], skip_special_tokens=True)
  return conclusion
  
def run_conclusions(df_val, model, tokenizer, df_train):
  for idx, row in df_val.iterrows:
    title = row['title']
    text = row['selftext']
    conclusion = generage_conclusion(title, text, model, tokenizer, df_train)
    print(conclusion) # or save result in table.
