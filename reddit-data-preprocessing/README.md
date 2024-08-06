# This directory contains code that processes the reddit data
* data obtained from [academic torrents](https://academictorrents.com/)
* no data is stored in this repo
* data are organized in RS-year-month.zst for posts(submissions) and RC-year-month.zst for comments

## code description
* decompress_write_par.py
  * unzip the .zst file
  * query keywords "copyright" and "regist", case-insensitive
  * output data in .json files

* combine_submissions.py
  * count repetitions in selftext and title
    * record "selftext" values. if "selftext" is empty, record "title" values
    * create dictionaries to save the repetitions in the same year-month or in all the files we selected
  * drop irrelevant columns
  * convert "awardings" from object to number
  * excel-compatible operations
    * fill empty "selftext" with the string "no text"
    * for len(selftext) > excel cell character limit, replace it with a hint of oversize and paste the url in the selftext.
    * for titles starts with "=", add a letter in the front to avoid excel mistaken it for functions (not resolvable by setting cell to "text" in "cell properties")
  * output in separate csv files by year-month
 
* hash_author.py
   * create a hash table to mask all author names
 
* process_authortime.py
   * convert author with hashed code
   * convert time to '%Y-%m-%d %H:%M:%S'
 
* subset.py
   * takes in a col_sel.txt file, which contains one column named "COLUMNS", and all the columns selected in order
   * subset the selected columns from each csv file.

* preprocessing steps
   1. run decompress_rewrite_par.py on a cluster to obtain the .json files. Manually scp and move the .json files in local directory register_csv_submissions/ (currently calling it this name because we are looking at "copyright registration")
   2. create output directory csv_out, under it, create directories: full_csv, converted_csv, and subset
   3. run combine_submissions.py to convert the .json files to .csv files.
   4. run hash_author.py to create a hash table
   5. use procee_authortime.py and the hash table created in the last process to convert author and time formats.
   6. manually modify the col_sel.txt file, run subset.py to create clear csv files.
 
* excel macros to make things easier (not shared here due to protection, but easy to generate with ChatGPT)
   * the macro to highlight keywords in selected columns
   * the macro for multiple choice.
