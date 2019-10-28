import numpy as np
import json
import os
from config import VCR_ANNOTS_DIR
import argparse

parser = argparse.ArgumentParser(description='Evaluate question -> answer and rationale')
parser.add_argument(
    '-preds',
    dest='preds',
    help='Location of predictions',
    type=str,
)

parser.add_argument(
    '-split',
    dest='split',
    default='val',
    help='Split you\'re using. Probably you want val.',
    type=str,
)

parser.add_argument(
    '-rationale',
    action="store_true",
    help='use rationale',
)

args = parser.parse_args()

preds= np.load(args.preds)
labels = []
with open(os.path.join(VCR_ANNOTS_DIR, '{}.jsonl'.format(args.split)), 'r') as f:
    for l in f:
        item = json.loads(l)
        if args.rationale:
            labels.append(item['rationale_label'])
        else:
            labels.append(item['answer_label'])

labels = np.array(labels)

print ('accuracy is : {}'.format(np.mean(labels == np.argmax(preds,1))))

