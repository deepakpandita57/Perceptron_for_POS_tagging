#!/usr/bin/python2

# Author: Deepak Pandita
# Date created: 15 Sep 2017

import json
import numpy as np

train_file = '/pos/train'
test_file = '/pos/test'
#file containing all tags
tag_file = 'All_tags'
max_iter = 8


all_tags = []
with open(tag_file) as tf:    
    all_tags = json.load(tf)


#Viterbi Algorithm
def viterbi(W,test_sequence,all_tags):
	best_tag_sequence = []
	viterbi_matrix = np.ones((len(all_tags),len(test_sequence)))*float('-inf')
	backpointer = np.zeros((len(all_tags),len(test_sequence)))
	for index,obs in enumerate(test_sequence):
		for tag_index,tag in enumerate(all_tags):
			if index==0:
				viterbi_matrix[tag_index,index] = 0
				backpointer[tag_index,index] = -1
			else:
				#scores = []
				b = W.get(obs+'_'+tag,-1e5)
				for prev_index,prev_tag in enumerate(all_tags):
					a = W.get(tag+'_'+prev_tag,-1e5)
					score = viterbi_matrix[prev_index,index-1]+a+b
					#scores.append(score)
					if(score > viterbi_matrix[tag_index,index]):
						viterbi_matrix[tag_index,index] = score
						backpointer[tag_index,index] = prev_index
				#viterbi_matrix[tag_index,index] = max(scores)	#get max score
				#backpointer[tag_index,index] = np.argmax(scores)		#get argmax
	
	#The start of backtrace = index of max probability in the last column of the probability matrix
	backtrace = np.argmax(viterbi_matrix[:,-1])
	
	for prev_best_tag in xrange(np.size(backpointer,axis=1),0,-1):
		best_tag_sequence.append(all_tags[int(backtrace)])
		backtrace = backpointer[int(backtrace),prev_best_tag-1]

	best_tag_sequence.reverse()
	return best_tag_sequence

def generate_features(X,Y):
	features = []
	i=0
	for x,y in zip(X,Y):
		features.append(x+'_'+y)
		if i!=0:
			features.append(y+'_'+Y[i-1])
		i = i + 1
	return features

#Read train file
print 'Reading file: '+train_file
f = open(train_file)
all_sentences = f.readlines()
all_sentence_words = []
all_sentence_tags = []
for line in all_sentences:
	tokens = line.strip().split(' ')[1:]
	words = [y for x,y in enumerate(tokens) if x%2 == 0]
	tags = [y for x,y in enumerate(tokens) if x%2 != 0]
	all_sentence_words.append(words)
	all_sentence_tags.append(tags)

#weights
W={}
Ws = []

#perceptron

print 'Running Perceptron...'
iter = 0
while (iter < max_iter):
	iter = iter + 1
	print 'Iteration: ' + str(iter)
	#incorrect_seq = 0
	
	
	for lineNum, words in enumerate(all_sentence_words):
		tags = all_sentence_tags[lineNum]
		
		#print words
		#print tags
	
		predicted_tags = viterbi(W,words,all_tags)
		#print predicted_tags
	
		if(tags!=predicted_tags):
			#incorrect_seq = incorrect_seq + 1
			prev_features = generate_features(words,tags)
			pred_features = generate_features(words,predicted_tags)
			for ys,vs in zip(prev_features,pred_features):
				if (ys!=vs):
					W[ys] = W.get(ys,0) + 1
					W[vs] = W.get(vs,0) - 1
	
		if lineNum%2000==0:
			print 'Processed '+str(lineNum)+' sentences'
	print 'Total sentences processed: '+str(lineNum)
	Ws.append(W)

def average(Ws):
	W={}
	K = len(Ws)
	for key in Ws[K-1].keys():
		sum = 0
		for i in range(0,K):
			sum+=Ws[i].get(key,0)
		W[key] = (float(sum) / K)
	return W

W = average(Ws)

#store weights
with open("weights", 'w') as wf:
	json.dump(W,wf)
	
#print len(W)
#Read test file
print 'Reading file: '+test_file

correct_tags = 0
total_tags = 0
with open(test_file) as tf:
	lineNum = 0
	for line in tf:
		tokens = line.strip().split(' ')[1:]
		words = [y for x,y in enumerate(tokens) if x%2 == 0]
		tags = [y for x,y in enumerate(tokens) if x%2 != 0]
		#Call viterbi algorithm
		best_tag_sequence = viterbi(W,words,all_tags)
		#print words
		#print tags
		#print best_tag_sequence
		for x in range(len(tags)):
			if tags[x]==best_tag_sequence[x]:
				correct_tags = correct_tags + 1
		total_tags = total_tags + len(tags)
		
		lineNum = lineNum + 1
		if lineNum%200==0:
			print 'Read '+str(lineNum)+' lines'
	print 'Total lines read: '+str(lineNum)

print 'Correct tags: '+str(correct_tags)+' Total tags: '+str(total_tags)
print 'Accuracy: ' + str(float(correct_tags)/total_tags)
print 'Done'