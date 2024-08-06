import multiprocessing as mp
import pandas as pd
import json
import zstandard as zstd
import re
import io
import time
import sys

def create_search(word_list):
	base = r'^{}'
	expr = '(?=.*{})'
	search_str = base.format(''.join(expr.format(w) for w in word_list))
	return search_str

def process_lines(lines, search_list, max_lines, result_queue, ignore_case = True, ignore_idx = -1):
#def process_lines(lines, search_list,  result_queue, ignore_case = True, ignore_idx = -1):
	"""
	Process a list of lines to search for a keyword and convert matching lines to JSON objects.
	"""
	line_list = []
	item_count = 0
	if ignore_case:
		search_str = create_search(search_list)
	else:
		if ignore_idx < 0:
			search_str = create_search(search_list)
	
		else:
			search_str = create_search(search_list[:ignore_idx])
			search_str_case = create_search(search_list[ignore_idx:])		

	for line in lines:
		if ignore_case:
			truth_search = re.search(search_str, line, re.IGNORECASE)
		elif ignore_idx < 0:
			truth_search = re.search(search_str, line)
		else: 
			truth_search = re.search(search_str, line, re.IGNORECASE)
			truth_search = truth_search and re.search(search_str_case, line)
	
		if truth_search:
			obj = json.loads(line)
			line_list.append(obj)
			item_count += 1
			if item_count == max_lines:
				break
			 
		 
	result_queue.put(line_list)

def worker(task_queue, result_queue, search_str, max_lines, ignore_case = True, ignore_idx = -1):
#def worker(task_queue, result_queue, search_str, ignore_case = True, ignore_idx = -1):
	"""
	Worker function to process lines in parallel.
	"""
	while True:
		lines = task_queue.get()
		if lines is None:# Sentinel value to stop the worker
			break
		process_lines(lines, search_str, max_lines, result_queue, ignore_case, ignore_idx)
#		process_lines(lines, search_str, result_queue, ignore_case, ignore_idx)


def write_to_csv(line_list, output_header, file_id):
	"""
	Write a list of JSON objects to a CSV file.
	"""
	df_out = pd.json_normalize(line_list)
	output_file = f"{output_header}_{file_id}.csv"
	df_out.to_csv(output_file, index=False)


def main(compressed, search_list, max_lines, output_header, max_window_size, ignore_case = True, ignore_idx = -1 , num_workers=4, chunk_size=1000):
#def main(compressed, search_list, output_header, max_window_size, ignore_case = True, ignore_idx = -1 , num_workers=4, chunk_size=1000):
	"""
	Main function to read the compressed file, distribute the work, and collect results.
	"""
	task_queue = mp.Queue()
	result_queue = mp.Queue()

	# Start worker processes
	processes = []
	for _ in range(num_workers):
		#p = mp.Process(target=worker, args=(task_queue, result_queue, search_list, max_lines, ignore_case, ignore_idx))
		p = mp.Process(target=worker, args=(task_queue, result_queue, search_list, ignore_case, ignore_idx))
		processes.append(p)
		p.start()

	# Decompress and read lines in chunks
	with open(compressed, "rb") as fi:
		dctx = zstd.ZstdDecompressor(max_window_size=max_window_size)
		stream_reader = dctx.stream_reader(fi)
		text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')

		lines = []
		for line in text_stream:
			lines.append(line)
			if len(lines) >= chunk_size:
				task_queue.put(lines)
				lines = []

		# Put any remaining lines in the queue
		if lines:
			task_queue.put(lines)

	# Add sentinel values to stop workers
	for _ in range(num_workers):
		task_queue.put(None)

	# Collect results from all workers
	all_results = []
	while any(p.is_alive() for p in processes) or not result_queue.empty():
		while not result_queue.empty():
			all_results.extend(result_queue.get())
	
#	# write the results to a json file
#	with open(output_header+'.json','w') as f:
#		json.dump(all_results, f, indent=4)

#	# Write results to multiple CSV files based on max_lines
	file_id = 0
	current_lines = []
	for obj in all_results:
		current_lines.append(obj)
		if len(current_lines) >= max_lines:
			write_to_csv(current_lines, output_header, file_id)
			current_lines = []
			file_id += 1

	# Write any remaining lines
	if current_lines:
		write_to_csv(current_lines, output_header, file_id)

	# Ensure all worker processes are finished
	for p in processes:
		p.join()

if __name__ == "__main__":
	#file_header = 'comments/RC_2023-10'
	file_header = str(sys.argv[1])
	compressed = file_header + '.zst' # Path to your zstandard compressed file
	search_list = ['copyright', 'regist'] # keywords	
	search_list2 = ['copyright', 'AI'] # another set of key words we searched	
	max_lines = 1000  # Maximum lines per CSV file
	output_header = 'csv_' + file_header 
	max_window_size = 2**31
	num_workers = 40 # Number of worker processes
	chunk_size = 1000 # Number of lines to read in each chunk

	output_header1 = 'register_' + output_header
	output_header2 = 'AI_' + output_header


	start_time = time.time()
	# search for 'copyright' and 'regist', ignoring cases
	main(compressed, search_list, max_lines, output_header1, max_window_size,  True,  -1, num_workers, chunk_size)
	#main(compressed, search_list, output_header1, max_window_size,  True,  -1, num_workers, chunk_size)

    # search for 'copyright' and 'AI', ignoring cases for 'copyright', all caps for 'AI'
	#main(compressed, search_list2, max_lines, output_header2, max_window_size, False, 1, num_workers, chunk_size)
	#main(compressed, search_list2, output_header2, max_window_size, False, 1, num_workers, chunk_size)
	end_time = time.time()
	elapsed_time = end_time - start_time
	print(f"Elapsed time: {elapsed_time} seconds")
