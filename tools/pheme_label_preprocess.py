"""to preprocess the Pheme dataset

Convert Rumor annotations into TrueRumor, FalseRumor, and UnverifiedRumor
This program should be worked on raw "Pheme" dataset
"""

import json
import os


def convert_annotations(annotation, string=True):
    """Convert rumour annotations into True, False, Unverified."""
    if 'misinformation' in annotation.keys() and 'true' in annotation.keys():
        if int(annotation['misinformation']) == 0 and int(annotation['true']) == 0:
            if string:
                label = "unverified"
            else:
                label = 2
        elif int(annotation['misinformation']) == 0 and int(annotation['true']) == 1:
            if string:
                label = "true"
            else:
                label = 1
        elif int(annotation['misinformation']) == 1 and int(annotation['true']) == 0:
            if string:
                label = "false"
            else:
                label = 0
        elif int(annotation['misinformation']) == 1 and int(annotation['true']) == 1:
            # print("OMG! They both are 1!")
            # print(annotation['misinformation'])
            # print(annotation['true'])
            label = None

    elif 'misinformation' in annotation.keys() and 'true' not in annotation.keys():
        # all instances have misinfo label but don't have true label
        if int(annotation['misinformation']) == 0:
            if string:
                label = "unverified"
            else:
                label = 2
        elif int(annotation['misinformation']) == 1:
            if string:
                label = "false"
            else:
                label = 0

    elif 'true' in annotation.keys() and 'misinformation' not in annotation.keys():
        # print('Has true not misinformation')
        label = None
    else:
        # print('No annotations')
        label = None

    return label



def get_label_dic(dataset_path):
    """Map root ids to specific labels."""
    label_dict = {}
    events = os.listdir(dataset_path)

    for event in events:
        if event[0] == '.':
            continue
        rumors_path = dataset_path + "\\" + event + "\\"+"rumours"
        rumors = os.listdir(rumors_path)
        for rumor in rumors:
            eid = str(rumor)
            # print(eid)
            if rumor[0] == ".":
                continue
            annotation = rumors_path + "\\" + rumor + "\\" + "annotation.json"
            with open(annotation, 'r', encoding='UTF-8') as f:
                annotation_dict = json.load(f)
                label = convert_annotations(annotation_dict)
                label_dict[eid] = str(label)

    return label_dict


def convert_labels(label_dict, data_path):
    """Revise labels in data file."""

    # input file
    fin = open(data_path, 'rt')
    # output file to write the result to
    fout = open("Pheme_label_All.txt", 'a')

    for line in fin:
        line = line.rstrip('\n')
        label, event, eid = line.split('\t')[0], line.split('\t')[1], line.split('\t')[2]
        if eid in label_dict.keys():
            label = label_dict[eid]
            info = label + '\t' + event + '\t' + eid + "\n"
            fout.write(info)

    fin.close()
    fout.close()


if __name__ == '__main__':
    base_path = "PHEME_veracity\\all-rnr-annotated-threads"     # path of dataset
    label_dict = get_label_dic(base_path)

    data_path = "data.label.txt"
    convert_labels(label_dict, data_path)
