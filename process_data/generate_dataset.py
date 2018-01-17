import csv
import json
import collections
import pandas as pd

def parse_csv(file_csv):
	task_ids = []
	task_texts = []
	task_tweet_id = []
	with open(file_csv, 'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',')
		
		for row in spamreader:
			task_ids.append(row[2])
			task_tweet_id.append(row[9])
			task_texts.append(row[10])
	return task_ids, task_texts, task_tweet_id

def parse_csv_task_runs(file_csv):
	task_ids = []
	task_labels = []
	task_tweet_ids = []
	task_info = []
	with open(file_csv, 'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',')
		
		for row in spamreader:
			task_info.append(row[4])
			task_ids.append(row[6])
			task_labels.append(row[10])
			task_tweet_ids.append(row[12])
	return task_ids, task_labels, task_tweet_ids, task_info

def parse_json(json_info):
	tweet_ids = []

	for row in json_info:
		jsonstring = json.loads(row)
		tweet_ids.append(jsonstring["tweet_id_tweet"])

	return tweet_ids

def read_id_label(file_id_label):
	task_runs_ids = []
	task_labels = []

	with open(file_id_label, 'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter='\t')

		for row in spamreader:
			task_runs_ids.append(row[0])
			task_labels.append(row[1])

	return task_runs_ids, task_labels

def label_text_unique(tasks,task_runs,criterio):
	label_text = []
	tweetid_label = []

	for task in tasks:
		count=0
		task_label_temp = ""
		for task_run in task_runs:
			#compare tweet id, if is equal then add the label to the tweet.
			if task_run[2] == task[2]:
				count+=1
				task_label_temp+= ',' + task_run[1]
		#if a tweet has more than 3 label then we use this heuristic for choose one label.
		if count >= 3:
			labels = task_label_temp.split(',')
			counter = collections.Counter(labels)
			#aca esta la etiqueta
			#print(counter.most_common(1)[0][0])
			#aca esta la frecuencia
			#print(counter.most_common(1)[0][1])
			if float(counter.most_common(1)[0][1])/float(len(labels)) >= criterio:
				label_text_aux = str(counter.most_common(1)[0][0]) + '\t' + (task[1])
				tweetid_label_aux = str(task[2]) + '\t' + str(counter.most_common(1)[0][0])
				label_text.append(label_text_aux)
				tweetid_label.append(tweetid_label_aux)

	return label_text, tweetid_label

def to_dataframe(label_texts):
	labels = []
	texts = []
	for row in label_texts:
		aux = row.split('\t')
		labels.append(aux[0])
		texts.append(aux[1])

	df = pd.DataFrame(dict(label=labels, tweet=texts))
	return df

if __name__ == "__main__":

	#se lee los task para obtener el tweet y su id
	task_ids, task_texts, task_tweet_id = parse_csv('terremoto-iquique-2014_task.csv')

	task_ids = task_ids[1:len(task_ids)]
	task_texts = task_texts[1:len(task_texts)]
	task_tweet_id = task_tweet_id[1:len(task_tweet_id)]

	#zip al the task elements in a list
	tasks = zip(task_ids,task_texts,task_tweet_id)

	#se lee task runs
	task_run_ids, task_run_labels, task_run_tweet_id, task_run_info = parse_csv_task_runs('terremoto-iquique-2014_task_run.csv')

	task_run_ids = task_run_ids[1:len(task_run_ids)]
	task_run_labels = task_run_labels[1:len(task_run_labels)]
	task_run_tweet_id = task_run_tweet_id[1:len(task_run_tweet_id)]
	task_run_info = task_run_info[1:len(task_run_info)]

	tweet_ids_json = parse_json(task_run_info)

	task_runs = zip(task_run_ids,task_run_labels,tweet_ids_json)	

	#se lee los task run donde cada tweet tiene al menos tres etiquetas 
	#task_runs_ids, task_labels = read_id_label('id-label.csv')
	#task_runs = zip(task_runs_ids,task_labels,task_tweet_id)

	label_texts, tweetid_label = label_text_unique(tasks,task_runs,0.5)

	print(len(label_texts))
	print(len(tweetid_label))


	df_label_texts = to_dataframe(label_texts)
	groups = df_label_texts.groupby('label')
	print(groups)

	with open('test_dataset_terremoto_iquique_2014.csv','w') as f:
		for row in label_texts:
			aux = row.split('\t')
			f.write(aux[0])
			f.write('\t')
			f.write(aux[1])
			f.write('\n')
		f.close()



	#with open('terremoto_iquique_2014_idtweet_label.csv','w') as f:
	#	for row in tweetid_label:
	#		aux = row.split('\t')
	#		f.write(aux[0])
	#		f.write('\t')
	#		f.write(aux[1])
	#		f.write('\n')












	