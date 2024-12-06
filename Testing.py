from CommunityDetection import CommunityDetection
from TFIDF import TF_IDF
import pickle
import numpy as np
import json

def evaluate(Community_Detection):
    with open('./eval_data.pkl', 'rb') as file:
        eval_data = pickle.load(file)

    with open('datasets/tag_id_name.json', "r") as file:
        tag_name_map_entries = json.load(file)
    tag_name_map = {int(entry['tag_id']):entry['tag_name'] for entry in tag_name_map_entries}

    results = {}
    for tag_id in eval_data.keys():
        tag_name = tag_name_map[tag_id] 
        recommended_tags,time,recommended_tag_ids = Community_Detection.community_detection(tag_name)
        results[tag_id] = recommended_tag_ids

    precision = 0
    for tag_id,recommended_tags in results.items():
        matched_tag_ids = sum([(True if int(rec_tag) in eval_data[tag_id] else False) for rec_tag in recommended_tags])
        print(matched_tag_ids)
        precision += matched_tag_ids/len(recommended_tags)
  
    precision = precision/len(results.keys())
    print('precision',precision)


    