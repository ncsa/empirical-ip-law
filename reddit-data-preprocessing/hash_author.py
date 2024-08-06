import pandas as pd
import random
import uuid

#in_dir = 'full_corrected_csv/'
in_dir = 'full_csv3/'

authors = []
dims = 0
for yr in ['2022','2023']:
    for mt in ['01','02','03','04',
               '05','06','07','08',
               '09','10','11','12']:
        file_header = 'RS_' + yr + '-' + mt
        fin = in_dir + file_header + '.csv'
        df = pd.read_csv(fin)
        authors = authors + df['author'].tolist()
        dims = dims + df.shape[0] 

authors = list(set(authors))
authors.sort()
random.Random(42).shuffle(authors)

with open('author_hash.csv','w') as f:
    for ii in range(len(authors)):
        rnd = random.Random()
        rnd.seed(ii)
        author_id=str(uuid.UUID(int=rnd.getrandbits(128), version=4))
        
        f.write(','.join([str(ii), authors[ii], author_id]))
        f.write('\n')

