import argparse
import os
import json
import csv
from tqdm import *
import random
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pickle
import ndjson
import base64
import sys
import numpy as np
from multiprocessing import Process, Queue,Manager
from nltk.wsd import lesk

parser = argparse.ArgumentParser() 
parser.add_argument(
    '-out',
    dest='out',
    type=str
)
parser.add_argument(
    '-split',
    dest='split',
    type=str
)
args = parser.parse_args()

csv.field_size_limit(sys.maxsize)
lemmatiser = WordNetLemmatizer()

data_folder = "../data/"

# Load classes
classes = ['__background__']
with open(os.path.join(data_folder, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        classes.append(object.split(',')[0].lower().strip())

# Load attributes
attributes = ['__no_attribute__']
with open(os.path.join(data_folder, 'attributes_vocab.txt')) as f:
    for att in f.readlines():
        attributes.append(att.split(',')[0].lower().strip())

# create output folder
os.mkdir('data_folder' + args.split + '_pickles_' + args.out)

# load the tsv file for object detections info
FIELDNAMES = ['image_id', 'num_boxes', 'objects' ]

imageid2det = {}
class_count = {}
with open(data_folder + 'new_tag_features_light_weight_{}.tsv'.format(args.split), "r") as tsv_in_file:
    reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
    for item in reader:
        image_id = args.split + '-' + item['image_id']
        obj_indices = np.frombuffer(base64.b64decode(item['objects']),dtype=np.int)
        num_boxes = int(item['num_boxes'])
        assert (num_boxes == len(obj_indices))
        obj_names_list = [classes[obj_idx+1] for obj_idx in obj_indices]
        
        # count the class occurence frequency
        for obj_name in obj_names_list:
            if obj_name not in class_count:
                class_count[obj_name] = 0
            class_count[obj_name] += 1
            
        imageid2det[image_id] = obj_names_list

vcr = open(data_folder + '{0}.jsonl'.format(args.split))
data = ndjson.load(vcr)

num_workers = 12 
manager = Manager()
annotid2det_q = manager.list([None] * num_workers)
question_annotid2detidx_q = manager.list([None] * num_workers)
answer_annotid2detidx_q = manager.list([None] * num_workers)
rationale_annotid2detidx_q = manager.list([None] * num_workers)

def worker(wid, class_count, annotid2det_q, question_annotid2detidx_q, answer_annotid2detidx_q, rationale_annotid2detidx_q):

    def get_word_det_similarity(word_synset, det_synsets):
        max_similarity = 0
        max_det_synset = None
        for det_synset in det_synsets:
            cur_similarity = word_synset.wup_similarity(det_synset)
            if cur_similarity > max_similarity:
                max_similarity = cur_similarity
                max_det_synset = det_synset
        return max_similarity, max_det_synset

    def synonym_match(word, det_obj_names, similarity_dict, sentence):
        word_match = False
        matched_det = None
        max_similarity_among_dets = 0
        word_lemma = lemmatiser.lemmatize(word,'n')
        word_sense = wordnet.synsets(word_lemma,'n')[0]
        word_sense_name = word_sense.name()
        matched_det_synset = None
        for det_idx, det_obj_name in enumerate(det_obj_names):
            if (word_sense_name, det_obj_name) not in similarity_dict: # calculate word - det class similarity, expensive
                if det_obj_name in class_synsets.keys():
                    max_similarity,max_det_synset = get_word_det_similarity(word_sense, class_synsets[det_obj_name])
                    similarity_dict[(word_sense_name, det_obj_name)] = (max_similarity,max_det_synset)
                else: # detection class that does not have noun synsets
                    similarity_dict[(word_sense_name, det_obj_name)] = (0,None) 
            word_det_similarity,max_det_synset = similarity_dict[(word_sense_name, det_obj_name)]
            if word_det_similarity >= similarity_threshold: 
                word_match = True
                if word_det_similarity > max_similarity_among_dets or (word_det_similarity == max_similarity_among_dets and word == det_obj_name):
                    matched_det = det_obj_name
                    matched_det_synset = max_det_synset
                    max_similarity_among_dets = word_det_similarity
             
        return word_match, matched_det, max_similarity_among_dets, word_sense, matched_det_synset

    def get_synset_idx(word, word_synset, synset_idx_dict):
        synset_name = word_synset.name()
        if (word,synset_name) not in synset_idx_dict:
            for idx,synset in enumerate(wordnet.synsets(word,'n')):
                synset_idx_dict[(word,synset.name())] = idx 
        return synset_idx_dict[(word,synset_name)]

    def process_sentence(sentence, gt_objects, ignore_set, det_obj_set, det_obj_names, det_obj_name_2_det_idx, similarity_dict, synset_idx_dict):
        word_idx_2_det_idx = {}
        word_2_det = {}
        remapped_sentence,new_word_idx_2_old, new_word_idx_2_det = remap_sentence_real_name(sentence, gt_objects)
        pos_tagged_sentence = nltk.pos_tag(remapped_sentence)
        num_matches = 0
        for word_idx in range(len(pos_tagged_sentence)):
            word = pos_tagged_sentence[word_idx][0].lower()
            tag = pos_tagged_sentence[word_idx][1]
            if word_idx in new_word_idx_2_det or len(wordnet.synsets(word, pos='n')) == 0:
                continue

            if tag != 'NN' and tag != 'NNS':
                continue
            if word in det_obj_set:
                word_match = True
                matched_det = word
                max_similarity_among_dets = 1
                word_synset = None
                matched_det_synset = None
            else:
                word_match, matched_det, max_similarity_among_dets, word_synset, matched_det_synset = synonym_match(word, det_obj_names, similarity_dict, remapped_sentence)

            # check match and count stats
            if (word_match and matched_det not in ignore_set and word not in ignore_set):
                if not word_synset: # obtained by direct match
                    word_synset_idx = -1
                    matched_det_synset_idx = -1
                else:
                    word_lemma = lemmatiser.lemmatize(word)
                    word_synset_idx = get_synset_idx(word_lemma, word_synset, synset_idx_dict)
                    matched_det_synset_idx = get_synset_idx(matched_det, matched_det_synset, synset_idx_dict)

                if (word, word_synset_idx,matched_det, matched_det_synset_idx) not in word_2_det:
                    word_2_det[(word, word_synset_idx,matched_det, matched_det_synset_idx)] = 0
                word_2_det[(word, word_synset_idx,matched_det, matched_det_synset_idx)] += 1
                old_word_idx = new_word_idx_2_old[word_idx]
                word_idx_2_det_idx[old_word_idx] = det_obj_name_2_det_idx[matched_det]
                num_matches += 1

            elif word == 'men' and 'man' in det_obj_names:
                word_lemma = 'men'
                word_synset_idx = 2
                matched_det = 'man'
                matched_det_synset_idx = 1

                if (word, word_synset_idx,matched_det, matched_det_synset_idx) not in word_2_det:
                    word_2_det[(word, word_synset_idx,matched_det, matched_det_synset_idx)] = 0
                word_2_det[(word, word_synset_idx,matched_det, matched_det_synset_idx)] += 1
                old_word_idx = new_word_idx_2_old[word_idx]
                word_idx_2_det_idx[old_word_idx] = det_obj_name_2_det_idx[matched_det]
                num_matches += 1
        return num_matches, word_2_det, word_idx_2_det_idx


    GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall','Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']
    # remap answer sentence back
    def remap_sentence_real_name(sentence, objects):
        remapped_sentence = []
        new_word_idx_2_old = {}
        new_word_idx_2_det = {}
        sen_idx_offset = 0 # old_sen_word_idx + offset = remapped_sen_word_idx

        # remap
        for word_idx in range(len(sentence)):
            cur_word = sentence[word_idx]

            if isinstance(cur_word, list):
                # for 1st detecton
                obj_type = objects[cur_word[0]]
                text_to_use = GENDER_NEUTRAL_NAMES[
                    cur_word[0] % len(GENDER_NEUTRAL_NAMES)] if obj_type == 'person' else obj_type
                remapped_sentence.append(text_to_use)
                new_word_idx_2_old[word_idx+sen_idx_offset] = word_idx
                new_word_idx_2_det[word_idx+sen_idx_offset] = cur_word[0]
                
                if len(cur_word) > 1:
                    # for 2nd to last-1 detection
                    for det_idx in range(1,len(cur_word)-1):
                        remapped_sentence.append(',')
                        
                        obj_type = objects[cur_word[det_idx]]
                        text_to_use = GENDER_NEUTRAL_NAMES[
                            cur_word[det_idx] % len(GENDER_NEUTRAL_NAMES)] if obj_type == 'person' else obj_type
                        remapped_sentence.append(text_to_use)

                        new_word_idx_2_old[word_idx+sen_idx_offset+det_idx*2-1] = word_idx
                        new_word_idx_2_old[word_idx+sen_idx_offset+det_idx*2] = word_idx
                        new_word_idx_2_det[word_idx+sen_idx_offset+det_idx*2] = cur_word[det_idx]
                        
                    # for last detection
                    if word_idx + 3 <= len(sentence)-1 and sentence[word_idx+1] == ',' and sentence[word_idx+2] == 'and' and isinstance(sentence[word_idx+3], list):   
                        # remap to ', {name}' since there is a ', and {tag}' following'
                        remapped_sentence.append(',')
                        obj_type = objects[cur_word[-1]]
                        text_to_use = GENDER_NEUTRAL_NAMES[
                            cur_word[-1] % len(GENDER_NEUTRAL_NAMES)] if obj_type == 'person' else obj_type
                        remapped_sentence.append(text_to_use)

                        new_word_idx_2_old[word_idx+sen_idx_offset+(len(cur_word)-1)*2-1] = word_idx
                        new_word_idx_2_old[word_idx+sen_idx_offset+(len(cur_word)-1)*2] = word_idx
                        new_word_idx_2_det[word_idx+sen_idx_offset+(len(cur_word)-1)*2] = cur_word[-1]
                        
                        sen_idx_offset += (len(cur_word) - 1) * 2 # each det except 1st one will be remapped to ', {name}', which increase offset by 2 for each of them
                    elif len(cur_word) == 2:
                        # remap to 'and {name}'
                        remapped_sentence.append('and')
                        obj_type = objects[cur_word[-1]]
                        text_to_use = GENDER_NEUTRAL_NAMES[
                            cur_word[-1] % len(GENDER_NEUTRAL_NAMES)] if obj_type == 'person' else obj_type
                        remapped_sentence.append(text_to_use)

                        new_word_idx_2_old[word_idx+sen_idx_offset+(len(cur_word)-1)*2-1] = word_idx
                        new_word_idx_2_old[word_idx+sen_idx_offset+(len(cur_word)-1)*2] = word_idx
                        new_word_idx_2_det[word_idx+sen_idx_offset+(len(cur_word)-1)*2] = cur_word[-1]

                        sen_idx_offset += (len(cur_word) - 1) * 2 # the dets will be mapped to {name} and {name}, which increase offset by 2
                    else:
                        # remap to ', and {name}'
                        remapped_sentence.append(',')
                        remapped_sentence.append('and')
                        obj_type = objects[cur_word[-1]]
                        text_to_use = GENDER_NEUTRAL_NAMES[
                            cur_word[-1] % len(GENDER_NEUTRAL_NAMES)] if obj_type == 'person' else obj_type
                        remapped_sentence.append(text_to_use)

                        new_word_idx_2_old[word_idx+sen_idx_offset+(len(cur_word)-1)*2-1] = word_idx
                        new_word_idx_2_old[word_idx+sen_idx_offset+(len(cur_word)-1)*2] = word_idx
                        new_word_idx_2_old[word_idx+sen_idx_offset+(len(cur_word)-1)*2+1] = word_idx
                        new_word_idx_2_det[word_idx+sen_idx_offset+(len(cur_word)-1)*2+1] = cur_word[-1]

                        sen_idx_offset += (len(cur_word) - 1) * 2 +1 # each det except 1st & last one will be remapped to ', {name}', which increase offset by 2 for each of them. The last one will be remapped to ', and {name}', which increase offset by 3
            else:
                remapped_sentence.append(cur_word)
                new_word_idx_2_old[word_idx+sen_idx_offset] = word_idx
                
        return remapped_sentence,new_word_idx_2_old, new_word_idx_2_det

    class_synsets = {}
    for word in class_count.keys():
        if len(wordnet.synsets(word,pos='n')) > 0:
            class_synsets[word] = [wordnet.synsets(word, pos='n')[0]] # only first one    

    annotid2det = {}
    question_annotid2detidx = {}
    answer_annotid2detidx = {}
    rationale_annotid2detidx = {}

    ignore_set = set()
    similarity_threshold = 0.95
    similarity_dict = {}
    synset_idx_dict = {}
    num_to_process = len(data) // num_workers
    start_idx = wid*num_to_process
    end_idx = (wid+1) * num_to_process
    if wid == num_workers - 1: # last process
        end_idx = len(data)
    for idx in tqdm(range(start_idx, end_idx)):
        sample = data[idx]
        image_id = sample['img_id']
        det_obj_names = imageid2det[image_id]

        annotid2det[sample['annot_id']] = [{},{}, {}, {}, {}, {}, {}, {}, {}] # 1 qestion, 4 answer, 4 rationale
        question_annotid2detidx[sample['annot_id']] = {}
        answer_annotid2detidx[sample['annot_id']] = [{},{}, {}, {}]
        rationale_annotid2detidx[sample['annot_id']] = [{},{}, {}, {}]
        det_obj_name_2_det_idx = {}
        for det_obj_idx, det_obj_name in enumerate(det_obj_names):
            if det_obj_name not in det_obj_name_2_det_idx:
                det_obj_name_2_det_idx[det_obj_name] = []
            det_obj_name_2_det_idx[det_obj_name].append(det_obj_idx)

        det_obj_set = set(det_obj_names)

        # for question
        num_matches, word_2_det, word_idx_2_det_idx = process_sentence(sample['question'], sample['objects'], ignore_set, det_obj_set, det_obj_names, det_obj_name_2_det_idx, similarity_dict, synset_idx_dict)
        if num_matches > 0:
            annotid2det[sample['annot_id']][0] = word_2_det 
            question_annotid2detidx[sample['annot_id']] = word_idx_2_det_idx

        for idx in range(4):
            # for answers
            num_matches, word_2_det, word_idx_2_det_idx = process_sentence(sample['answer_choices'][idx], sample['objects'], ignore_set, det_obj_set, det_obj_names, det_obj_name_2_det_idx, similarity_dict,synset_idx_dict)
            annotid2det[sample['annot_id']][idx+1] = word_2_det 
            answer_annotid2detidx[sample['annot_id']][idx] = word_idx_2_det_idx

            # for rationales
            num_matches, word_2_det, word_idx_2_det_idx = process_sentence(sample['rationale_choices'][idx], sample['objects'], ignore_set, det_obj_set, det_obj_names, det_obj_name_2_det_idx, similarity_dict, synset_idx_dict)
            annotid2det[sample['annot_id']][idx+5] = word_2_det 
            rationale_annotid2detidx[sample['annot_id']][idx] = word_idx_2_det_idx
                
    print ('worker ' + str(wid) + 'count done')

    annotid2det_q[wid] = annotid2det
    question_annotid2detidx_q[wid] = question_annotid2detidx
    answer_annotid2detidx_q[wid] = answer_annotid2detidx 
    rationale_annotid2detidx_q[wid] = rationale_annotid2detidx 

jobs = []
for wid in range(num_workers):
    p = Process(target=worker, args=(wid, class_count, annotid2det_q, question_annotid2detidx_q, answer_annotid2detidx_q, rationale_annotid2detidx_q))
    jobs.append(p)
    p.start()
for p in jobs:
    p.join()

print ('all worker joined, start reducing')
def reduce(obj, out):
    agg_dict = {}
    agg_set = set()
    for cur_worker_item in obj:
        if cur_worker_item != None:
            if isinstance(cur_worker_item, dict):
                for key in cur_worker_item.keys():
                    if key not in agg_dict:
                        agg_dict[key] = cur_worker_item[key]
                    else:
                        agg_dict[key] += cur_worker_item[key]
            else:
                for item in cur_worker_item:
                    agg_set.add(item)
    with open(data_folder + args.split + '_pickles_' + args.out + '/' + out, 'wb') as f:
        if len(agg_dict) > 0:
            pickle.dump(agg_dict, f)
        else:
            pickle.dump(agg_set, f)

reduce(annotid2det_q, 'annotid2det.pkl')
reduce(question_annotid2detidx_q, 'question_annotid2detidx.pkl')
reduce(answer_annotid2detidx_q, 'answer_annotid2detidx.pkl')
reduce(rationale_annotid2detidx_q, 'rationale_annotid2detidx.pkl')

