from os import listdir
from os.path import isfile, join
from os.path import exists
import pandas as pd
import os


def five_class_setup():

    negative_dir = 'Z:/Lymphoma_UW_Retrospective/Data/mips/Group_1_2_3_curated'
    positive_dir = 'Z:/Lymphoma_UW_Retrospective/Data/mips/Group_4_5_curated'

    # gets all the file names in and puts them in a list
    neg_files = [f for f in listdir(negative_dir) if isfile(join(negative_dir, f))]
    pos_files = [f for f in listdir(positive_dir) if isfile(join(positive_dir, f))]

    if "Thumbs.db" in neg_files:
        neg_files.remove("Thumbs.db")
    if "Thumbs.db" in pos_files:
        pos_files.remove("Thumbs.db")

    all_files = neg_files + pos_files

    report_direct = 'Z:\Lymphoma_UW_Retrospective\Reports'
    reports_1 = pd.read_csv(os.path.join(report_direct, 'ds1_findings_and_impressions_wo_ds_more_syn.csv'))
    reports_2 = pd.read_csv(os.path.join(report_direct, 'ds2_findings_and_impressions_wo_ds_more_syn.csv'))
    reports_3 = pd.read_csv(os.path.join(report_direct, 'ds3_findings_and_impressions_wo_ds_more_syn.csv'))
    reports_4 = pd.read_csv(os.path.join(report_direct, 'ds4_findings_and_impressions_wo_ds_more_syn.csv'))
    reports_5 = pd.read_csv(os.path.join(report_direct, 'ds5_findings_and_impressions_wo_ds_more_syn.csv'))

    data_with_labels = pd.DataFrame(columns=['image_id','label'])
    i = 0
    missing_reports = 0
    for file in neg_files:

        file_check = file[0:13]

        if reports_1['id'].str.contains(file_check).any():
            data_with_labels.loc[i] = [file, 1]
        elif reports_2['id'].str.contains(file_check).any():
            data_with_labels.loc[i] = [file, 2]
        elif reports_3['id'].str.contains(file_check).any():
            data_with_labels.loc[i] = [file, 3]
        elif reports_4['id'].str.contains(file_check).any():
            data_with_labels.loc[i] = [file, 4]
        elif reports_5['id'].str.contains(file_check).any():
            data_with_labels.loc[i] = [file, 5]
        else:
            #print("Not in text files")
            missing_reports += 1
            i = i - 1

        i += 1

    #print(data_with_labels)
    #print(f"Missing reports: {missing_reports}")

    return data_with_labels
