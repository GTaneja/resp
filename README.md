# resp
# Define constants
in_dir=
out_dir=
raw_resume_dir=
parsed_resume_dir=
section_ident_resume_dir=
raw_tale_data_dir=
processed_tale_data_dir=
mapping_dir=
lst_dir=
ner_dir=
intermediate_dir=

# List of files to be included
import parsing_helper_functions as parser
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Structure:
#  file parsing_helper_functions.py will contain 2 functions
#         1. to_txt() will invoke the Java code to parse files from doc,docx and pdf to text
#            Inputs: 
#               (a) indir: filepath(in_dir,raw_resume_dir,"/")
#               (b) outdir: filepath(out_dir,parsed_resume_dir,"/")
#               (c) verbose : Logical input, show progress while code runs if True
#               (d) qc_check : Logical input, return a dictionary with {file_path,number of words} and display histogram of word count
#            Outputs:                
#               (a) Print:
#                       no. of files parsed successfully
#                       % of files passed successfully
#                       Histogram of Word Count ( IF qc_check argument was pssed as True)   
#        2. section_identifier() will invoke the Java copde that identifies Section Headers and will write the output in location provided
#           Inputs:
#               (a) indir: filepath(out_dir,parsed_resume_dir,"/")
#               (b) outdir: filepath(out_dir,section_ident_resume_dir,"/")
#               (c) verbose : Logical input, show progress while code runs if True
#               (d) qc_check : Logical input, return a dictionary with {file_path,word count(wc)in Summary, wc_Education.. so on} and display histogram of word count
#            Outputs:                
#               (a) Print:
#                       no. of files parsed successfully
#                       % of files passed successfully
#                       Histogram of Word Count byeach Section ( IF qc_check argument was pssed as True)   
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import taleo_data_processing as taleo
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Structure:
#  file tale_data_processing.py will contain all the code required to convert the raw taleo data to processes Taleop data. This
#  will include all the mapping, cleaning, filtering to be performed.
#  Functions:
#    1. process_data()
#       Inputs:
#          (a) indir: filepath(in_dir,raw_tale_data_dir)
#          (b) outdir: filepath(out_dir,processed_tale_data_dir)
#          (c) verbose: Logical input, show progress while code runs if True
#          (d) mapping_files: filepath(in_dir,mapping_dir)
#        Output:
#            (a) taleo_data: Data Frame 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import ner
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Structure:
#  file ner.py will include 13 functions, each one identifying occurance of a list of words/phrases in the provided resumes
#  Functions:
#    1.employer_recognition()
#    2.university_recognition()
#    3.degree_recognition()
#    4.major_recognition()
#    5.skill_recognition()
#    6.hobbies_recognition()
#    7.language_recognition()
#    8.certifiation_recognition()
#    9.candidate_name_recognition()
#    10.candidate_email_recognition()
#    11.candidate_contact_recognition()
#    12.candidate_address_recognition()
#    13.candidate_gender_recognition()
#        Inputs:
#            (a) indir1: filepath(out_dir,section_ident_resume_dir,"/")
#            (b) indir2: filepath(out_dir,lst_dir,"/")
#            (c) removal: logical argument, 
#                    if True then create a resume with all recognized entities removed from the text
#                    if False then tag each occrance of entity with <entity>entity name</entity>
#            (d) df: logical argument,
#                    if True then return the recognized entitities in a data frame like:
#                        File_Path,Employer1,Employer2,...Employer10,Univ1,Univ2,...,Univ10.. and so on
#                    if False then do nothing
#            (e) qc: logical argumenr, if True, display histogram of word counts for each entity type 
#            (f) outdir: filepath(out_dir,ner_dir)
#            (g) verbose : Logical input, show progress while code runs if True
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


import custom_feature_builder as fb
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Structure:
#  file feature_builder.py will include several functions,one for each customized feature we want to build
#  Inputs:
#    (a) indir: filepath(out_dir,section_ident_resume_dir)
#    (b) outdir: filepath(out_dir,features_csv) features_csv will contain two columns file_path, feature. If
#        outdir=NULL, then return a data frame with the 2 columns file_path and feature_value
#  Functions: some examples are
#        gen_skills()
#        gen_major()
#        gen_degree()
#        gen_employer_industry() # if worked at a finance frim, tech firm, fmcg etc.
#        gen_industry_score() # score in finanace, tech etc based on resume content
#        gen_years_of_employment()
#        gen_currently_employed()
#        gen_years_in_academia()
#        .... and so on
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import tfidf_pipeline as tfidfp
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Structure:
#  file tfidf_pipeline.py will generate TFIDF corpus   using gensim library in python
#  Inputs:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
