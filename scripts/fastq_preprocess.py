import pandas as pd
import os
import time
from dotmap import DotMap
from pathlib import Path
import math
import numpy as np
from cutadapt.adapters import AdapterParser
from cutadapt.seqio import Sequence
from scripts import feature_util


#########################################################################################
# 1 - Create the csv file and fill the task that has ben done.
def create_csv(config):
    print('\n\n######################################')
    print('1. Reading report.csv')

    if not os.path.exists(config.fastq_path + 'report.csv'):
        print('Didn\'t found report.csv file -> creating new file')
        # Comment: wt_1 fastq file was not been supplied correctly by the DepHF authors.
        col_names = ['background_1', 'background_2', 'hf_1', 'hf_2', 'esp_1', 'esp_2', 'wt_2']

        raw_names = ['split_file', 'masked_N', 'sort_and_count', 'drop_scaffold', 'unify_dataframes', 'create_fasta',
                     'align_to_genome', 'split_valid_df', 'extract_target', 'unite_target_df', 'calc_efficiency'
                     , 'write_to_final_table']
        report_df = pd.DataFrame(columns=col_names, index=raw_names)
        report_df[:] = 0

        # Disable some of the functions for bg's files
        report_df['background_1']['calc_efficiency'] = -1
        report_df['background_2']['calc_efficiency'] = -1
        report_df['background_1']['write_to_final_table'] = -1
        report_df['background_2']['write_to_final_table'] = -1

        report_df.to_csv(config.fastq_path + 'report.csv')

    else:
        print('Loading report.csv ......')
        report_df = pd.read_csv(config.fastq_path + 'report.csv')
        report_df.set_index('Unnamed: 0', inplace=True)

    print(report_df)
    return report_df


#########################################################################################
# 2 - splitting the fastq files for easier read with pd
# Comment: The original files can be deleted after finishing this stage for memory cleanup.
def split_files(config, report_df):
    print('\n\n######################################')
    print('2. splitting files')
    split_files_list = report_df.loc['split_file', :]
    split_files_dir = config.fastq_path + 'split_files/'
    split_size = 40000000  # 40 million rows

    if not os.path.exists(config.fastq_path + 'split_files'):
        print(f'Creating new directory - {split_files_dir}')
        os.mkdir(split_files_dir)

    for file_name, val in split_files_list.iteritems():
        if val == 0:
            print(f'splitting {file_name}')
            file = open(config.fastq_path + 'original_files/' + file_name + '.fastq', 'r')
            new_file_ind = 0
            new_file = open(split_files_dir + file_name + '_0.fastq', 'w')
            for ind, line in enumerate(file):
                if ind % 5000000 == 0:
                    print(ind)
                if ind % split_size == 0:
                    if new_file_ind != 0:
                        new_file.close()
                    new_file = open(split_files_dir + file_name + f'_{new_file_ind}.fastq', 'w')
                    new_file_ind += 1
                new_file.write(line)

            new_file.close()
            report_df.loc['split_file', file_name] = 1
            report_df.to_csv(config.fastq_path + 'report.csv')
        else:
            print(f'{file_name} already been split')


#########################################################################################
# 3 - Masking low quality reads with N and dropping reads with N in the barcode and more than 4 N's
# Comment: The original split files can be deleted after finishing this stage for memory cleanup.

def findN(seq, quality):
    new_seq = ''
    for s, q in zip(seq, quality):
        quality_val = ord(q)
        if quality_val <= 42:
            new_seq += 'N'
        else:
            new_seq += s
    return seq

def drop_N(config, report_df):
    print('\n\n######################################')
    print('3. Dropping low quality reads')
    mask_files_list = report_df.loc['masked_N', :]

    for file_name, val in mask_files_list.iteritems():
        if val == 0:
            file_index = 0
            while os.path.exists(config.fastq_path + 'split_files/' + file_name + f'_{file_index}.fastq'):
                file_path = config.fastq_path + 'split_files/' + file_name + f'_{file_index}.fastq'
                with open(file_path) as file:
                    masked_file = open(config.fastq_path + 'split_files/' + file_name + f'_{file_index}_masked.fastq', 'w')
                    file_index += 1
                    print(f'masking file {file_path}')
                    for ind, line in enumerate(file):
                        if ind%1000000 == 0:
                            print(ind)
                        if ind%4 == 0:
                            desc1 = line
                        elif ind%4 == 1:
                            seq = line
                        elif ind%4 == 3:
                            quality = line
                            seq = findN(seq, quality)
                            N_count = seq.count('N')
                            N_pos = seq.find('N')
                            if N_pos != -1:
                                if N_pos < 20 or N_count >=4:
                                    continue
                            masked_file.write(desc1)
                            masked_file.write(seq)
                    masked_file.close()
            report_df.loc['masked_N', file_name] = 1
            report_df.to_csv(config.fastq_path + 'report.csv')
        else:
            print(f'{file_name} already been dropped')


#########################################################################################
# 4- sort the files into dataframes
# Comment: The masked files can be deleted after finishing this stage for memory cleanup.

def fq2df(config, report_df):
    print('\n\n######################################')
    print('4. Sorting files into dataframes')

    sort_and_count_list = report_df.loc['sort_and_count', :]

    dataframes_path = config.fastq_path + 'dataframes/'
    if not os.path.exists(dataframes_path):
        os.mkdir(dataframes_path)

    for file_name, val in sort_and_count_list.iteritems():
        if val == 0:
            file_index = 0
            while os.path.exists(config.fastq_path + 'split_files/' + file_name + f'_{file_index}_masked.fastq'):
                file_path = config.fastq_path + 'split_files/' + file_name + f'_{file_index}_masked.fastq'
                print(f'fq2df on file:' + file_name + f'_{file_index}_masked.fastq')
                df = pd.read_csv(file_path, header=None)

                df.reset_index(inplace=True, drop=True)
                df_read = df[df.index % 2 == 1]
                df_read.columns = ['read']
                df_read.reset_index(inplace=True, drop=True)
                df_read.columns = ['read']
                df_unique = df_read.read.value_counts().reset_index()
                df_unique.columns = ['read', 'counts']
                df_unique.to_pickle(dataframes_path + f'{file_name}_{file_index}_df.pkl')
                file_index += 1

            report_df.loc['sort_and_count', file_name] = 1
            report_df.to_csv(config.fastq_path + 'report.csv')
        else:
            print(f'{file_name} has already been sorted into dataframe')

#########################################################################################
# 5 - delete all the sequences with wrong scaffold
def drop_scaffold(config, report_df):
    print('\n\n######################################')
    print('5. Dropping scaffold')

    drop_scaffold_list = report_df.loc['drop_scaffold', :]
    dataframes_path = config.fastq_path + 'dataframes/'
    scaffold_sequence = 'GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGCTTTTT'

    for file_name, val in drop_scaffold_list.iteritems():
        if val == 0:
            file_index = 0
            while os.path.exists(config.fastq_path + 'dataframes/' + file_name + f'_{file_index}_df.pkl'):
                file_path = config.fastq_path + 'dataframes/' + file_name + f'_{file_index}_df.pkl'
                print(f'Dropping scaffold in file: {file_path}')
                scaffold_df = pd.read_pickle(file_path)
                scaffold_df = scaffold_df[scaffold_df['read'].str.find(scaffold_sequence) == 20]
                gRNAs_line = scaffold_df.apply(lambda x: x['read'][:20], axis =1)
                read_sequence = scaffold_df.apply(lambda x: x['read'][101:], axis =1)
                scaffold_df.drop('read', axis='columns', inplace=True)
                scaffold_df['gRNA'] = gRNAs_line
                scaffold_df['read_sequence'] = read_sequence
                scaffold_df.to_pickle(dataframes_path + f'{file_name}_{file_index}_scaffold_df.pkl')
                file_index += 1

            report_df.loc['drop_scaffold', file_name] = 1
            report_df.to_csv(config.fastq_path + 'report.csv')
        else:
            print(f'{file_name} scaffold has already been dropped')

#########################################################################################
# 6 - unite all processed dataframes
def unite_dataframes(config, report_df):
    print('\n\n######################################')
    print('6. Uniting scaffold dataframes')
    unify_dataframes_list = report_df.loc['unify_dataframes', :]
    dataframes_path = config.fastq_path + 'dataframes/'

    for file_name, val in unify_dataframes_list.iteritems():
        if val == 0:
            file_index = 0
            unite_df = pd.DataFrame(columns=['counts', 'gRNA', 'read_sequence'])
            while os.path.exists(config.fastq_path + 'dataframes/' + file_name + f'_{file_index}_scaffold_df.pkl'):
                file_path = config.fastq_path + 'dataframes/' + file_name + f'_{file_index}_scaffold_df.pkl'
                print(file_path)
                df_part = pd.read_pickle(file_path)
                unite_df = pd.concat([unite_df, df_part])
                start = time.time()
                unite_df = unite_df.groupby(['gRNA', 'read_sequence'], as_index=False).agg('sum')
                end = time.time()
                print('time: {}'.format(end-start))
                file_index += 1

            unite_df.to_pickle(dataframes_path + f'{file_name}_unified_df.pkl')

            report_df.loc['unify_dataframes', file_name] = 1
            report_df.to_csv(config.fastq_path + 'report.csv')

        else:
            print(f'{file_name} scaffold files has already been united')


#########################################################################################
# 7 - Create fasta files of the barcode gRNA from the dataframes
def write_fasta(x, fasta_file):
    print('>read_{}|{}'.format(x['index'], x['counts']), x['gRNA'][1:20], file=fasta_file, sep='\n')


def create_fasta(config, report_df):
    print('\n\n######################################')
    print('7. Creating fasta files')

    create_fasta_list = report_df.loc['create_fasta', :]
    fasta_path = config.fastq_path + 'fasta_files/'
    if not os.path.exists(fasta_path):
        os.mkdir(fasta_path)

    for file_name, val in create_fasta_list.iteritems():
        if val == 0:
            print(f'Creating {file_name}.fa')
            file_path = config.fastq_path + f'dataframes/{file_name}_unified_df.pkl'
            df_unique = pd.read_pickle(file_path)
            fasta_file = open(fasta_path + f'{file_name}.fa', 'w')
            df_unique.reset_index().apply(lambda x: write_fasta(x, fasta_file), axis=1)
            fasta_file.close()

            report_df.loc['create_fasta', file_name] = 1
            report_df.to_csv(config.fastq_path + 'report.csv')

        else:
            print(f'{file_name} fasta file has already been created')

#########################################################################################
# 8 - Analyze the reference genome
def get_gRNA_seq(row):
    gRNA = row['Seq'][:20]
    return gRNA


def get_pos(row):
    pos = row['Seq'].rfind(row['gRNA_seq'])
    return pos


def get_xfix(row):
    suffix_start = row['target_pos'] + 23
    suffix_end = suffix_start + 10
    prefix_start = row['target_pos'] - 10
    prefix_end = row['target_pos']
    suffix = row['Seq'][suffix_start:suffix_end]
    prefix = row['Seq'][prefix_start:prefix_end]
    return prefix, suffix


def write_ref(row, ref_file):
    print('>{}|{}|{}'.format(row['target_gRNA_PAM'], row['prefix'], row['suffix']), row['gRNA_seq'],file=ref_file, sep='\n')


def analyze_ref(config):
    print('\n\n######################################')
    print('8. Analyzing reference file')
    ref_path = config.fastq_path + 'ref/'
    if not os.path.exists(ref_path):
        os.mkdir(ref_path)

    df_path = ref_path + 'df_ref_gRNA_choosen.pkl'
    if os.path.exists(df_path):
        print('Analyzed file already exists')
        df_ref_gRNA_choosen = pd.read_pickle(df_path)
    else:
        opt = DotMap()
        opt.path = Path('data/').expanduser()
        opt.df_ref_gRNA = opt.path.joinpath('suplementry1.csv')
        df_ref_gRNA_name = pd.read_csv(opt.df_ref_gRNA)

        df_ref_gRNA_name['gRNA_seq'] = df_ref_gRNA_name.apply(lambda x: get_gRNA_seq(x), axis=1)
        df_ref_gRNA_name['target_pos'] = df_ref_gRNA_name.apply(lambda x: get_pos(x), axis=1)
        df_ref_gRNA_name[['prefix', 'suffix']] = df_ref_gRNA_name.apply(lambda x: pd.Series(get_xfix(x)), axis=1)

        # exc = ['AGACAGTAGCCAAACACCC',
        #        "TGATATCGTGGTTCCTGGG",
        #        "TAAGTCAGTGGAAAGAAAG",
        #        "GTGATGGTCTTACCAGTCA",
        #        "CACTCACTCTTCTTGCAGG"]
        exc = []
        df_ref_gRNA_choosen = df_ref_gRNA_name[~df_ref_gRNA_name.gRNA_seq.apply(lambda x: x[1:20]).isin(exc)].copy()
        df_ref_gRNA_choosen['target_gRNA_PAM'] = df_ref_gRNA_choosen.apply(
            lambda x: x['Seq'][x['target_pos']:x['target_pos'] + 23].upper(), axis=1)
        df_ref_gRNA_choosen.to_pickle(df_path)

    gRNA_name_path = config.fastq_path + 'ref/gRNA_Name.fa'
    if not os.path.exists(gRNA_name_path):
        gRNA_name_file = open(gRNA_name_path, 'w')
        df_ref_gRNA_choosen.apply(lambda x: write_ref(x, gRNA_name_file), axis=1)
        gRNA_name_file.close()

    return df_ref_gRNA_choosen

#########################################################################################
# 9 - Aligning to the reference genome file
def align_to_genome(config, report_df):
    # Currently this stage is done partially manually
    # instructions:
    #   1. first we need to align the reference genome file. run the following script from the fastq_files/ref dir:
    #      bwa index -a bwtsw gRNA_Name.fa (Need to be done only once)
    #   2. now we need to run the following script to every file in the fastq_files/fasta_files dir from fastq_files/ref dir (it may take several minutes for each file)
    #      bwa aln -t 20 -n 0 -o 0 -l 19 -k 0 -d 1 -i 1 -O 5 -E 3 -N ../ref/gRNA_Name.fa <fasta file path> | bwa samse ../ref/gRNA_Name.fa - <fasta file
    #      path> | samtools view - | awk ' BEGIN { OFS="@"; }{print $1,$2,$3,$4,$6,$10,$13,$14,$15,$16,$17,$18,$19,$20} ' > <fasta file name>_aligned.sam
    #   3. download the out files to your computer under aligned_files

    print('\n\n######################################')
    print('9. Aligning to genome')
    valid_dir_path = config.fastq_path + 'gRNA_valid_df/'
    if not os.path.exists(valid_dir_path):
        os.mkdir(valid_dir_path)

    align_to_genome_list = report_df.loc['align_to_genome', :]
    column_names = ['read_index', 'FLAG', 'RNAME', 'POS', 'CIAGR', 'SEQ',
                    'EditDistance', 'No.BestHits', 'No.SuboptimalHits', 'No.Mismatches',
                    'No.GapOpen', 'No.GapExtension', 'MismathPosBase', 'AlternativeHits']

    for file_name, val in align_to_genome_list.iteritems():
        if val == 0:
            print(f'Aligning {file_name}')
            unique_df = pd.read_pickle(config.fastq_path + f'dataframes/{file_name}_unified_df.pkl')
            aligned_df = pd.read_csv(config.fastq_path + f'aligned_files/{file_name}_aligned.sam', header=None, delimiter='@',names=column_names)
            df_merge_gRNA_Name = pd.concat(
                [aligned_df.loc[:, ['read_index', 'FLAG', 'RNAME']], unique_df.reset_index(drop=True)],
                axis=1).sort_values(by='counts', ascending=False)
            df_gRNA_valid = df_merge_gRNA_Name[df_merge_gRNA_Name.FLAG == 0]
            df_gRNA_valid.to_pickle(valid_dir_path + f'{file_name}_valid_df.pkl')

            report_df.loc['align_to_genome', file_name] = 1
            report_df.to_csv(config.fastq_path + 'report.csv')

        else:
            print(f'{file_name} is already aligned')

#########################################################################################
# 10 - split valid data frame to make the processing easier for the cpu
def split_valid_df(config, report_df):
    print('\n\n######################################')
    print('10. Splitting valid_df')

    valid_dir_path = config.fastq_path + 'gRNA_valid_df/'

    split_valid_df_list = report_df.loc['split_valid_df', :]

    for file_name, val in split_valid_df_list.iteritems():
        if val == 0:
            print(f'Splitting {file_name}_valid_df.pkl')
            valid_df = pd.read_pickle(valid_dir_path + f'{file_name}_valid_df.pkl')
            num_partitions = math.ceil(len(valid_df) / 20000)  # number of partitions to split dataframe
            df_split = np.array_split(valid_df, num_partitions)
            for ind, df in enumerate(df_split):
                df.to_pickle(valid_dir_path + f'{file_name}_valid_df_{ind}.pkl')

            report_df.loc['split_valid_df', file_name] = 1
            report_df.to_csv(config.fastq_path + 'report.csv')
        else:
            print(f'{file_name}_valid_df.pkl has already been split')

#########################################################################################
# 11 - Extracting the target sequence using cutadapt
def get_read_info(row, df_ref_gRNA_choosen):
    prefix_range = ()
    suffix_ragne = ()
    target_range = ()
    read_prefix_seq = ''
    read_suffix_seq = ''
    read_gRNA_PAM = ''
    seq_101_150 = row.read_sequence
    read_seq = ('read_ing',seq_101_150)
    is_mis_synthesis = -1
    is_edited = -1
    lst_split = row.RNAME.split('|')
    designed_gRNA_PAM = lst_split[0]
    designed_prefix = lst_split[1]
    designed_suffix = lst_split[2]
    read_gRNA = '{}|{}'.format(row.read_index,designed_gRNA_PAM)
    # =====================
    # -a 3'
    a = []
    # -b both
    b = []
    # -g 5'
    g = ['gRNA_prefix_suffix={}...{}'.format(designed_prefix, designed_suffix)]
    adapter_parser = AdapterParser(
        colorspace=None,
        max_error_rate=0.2,
        min_overlap=17,
        read_wildcards=False,
        adapter_wildcards=False,
        indels=True)
    read = Sequence(name=read_seq[0], sequence=read_seq[1])
    adapter = adapter_parser.parse_multi(a, b, g)[0]

    # find the region of the prefix adapter and the suffix adapter
    r = adapter.match_to(read)
    # check for indels / synthesis errors
    if r is not None:
        prefix_range = (r.front_match.rstart,r.front_match.rstop,r.front_match.errors)
        suffix_ragne = (r.back_match.rstart + r.front_match.rstop,r.back_match.rstop + r.front_match.rstop,r.back_match.errors)
        target_range = (r.front_match.rstop, r.back_match.rstart +r.front_match.rstop)
        read_prefix_seq = seq_101_150[prefix_range[0]:prefix_range[1]]
        read_suffix_seq = seq_101_150[suffix_ragne[0]:suffix_ragne[1]]
        read_gRNA_PAM = seq_101_150[target_range[0]:target_range[1]]
        if designed_gRNA_PAM == read_gRNA_PAM:
            is_mis_synthesis = 0
            is_edited = 0
        elif read_gRNA_PAM in df_ref_gRNA_choosen.target_gRNA_PAM.values:
            is_mis_synthesis = 1
        else:
            is_mis_synthesis = 0
            is_edited = 1
    return ({'read_gRNA': read_gRNA, 'designed_gRNA_seq': designed_gRNA_PAM[:20], 'prefix_range': str(prefix_range),
            'suffix_ragne': str(suffix_ragne), 'target_range': str(target_range),'read_prefix_seq': read_prefix_seq,
            'read_gRNA_PAM': read_gRNA_PAM,'read_suffix_seq': read_suffix_seq, 'is_edited': is_edited,
            'is_mis_synthesis': is_mis_synthesis})

def extract_targets(config, report_df):
    print('\n\n######################################')
    print('11. Extracting the target sequence using cutadapt')
    extract_target_list = report_df.loc['extract_target', :]

    gRNA_target_dir_path = config.fastq_path + 'gRNA_target/'
    valid_dir_path = config.fastq_path + 'gRNA_valid_df/'

    if not os.path.exists(gRNA_target_dir_path):
        os.mkdir(gRNA_target_dir_path)

    df_ref_gRNA_choosen = pd.read_pickle(config.fastq_path + 'ref/df_ref_gRNA_choosen.pkl')

    for file_name, val in extract_target_list.iteritems():
        if val == 0:
            file_ind = 0
            file_path = valid_dir_path + f'{file_name}_valid_df_{file_ind}.pkl'
            while os.path.exists(file_path):
                print(f'Processing {file_path}')
                df_gRNA_valid = pd.read_pickle(file_path)
                target_df = df_gRNA_valid.apply(lambda x:pd.Series( get_read_info(x, df_ref_gRNA_choosen) ),axis=1 )
                target_df.columns = ['read_gRNA', 'designed_gRNA_seq', 'prefix_range', 'suffix_ragne',
                              'target_range', 'read_prefix_seq', 'read_gRNA_PAM',
                              'read_suffix_seq', 'is_edited', 'is_mis_synthesis']
                target_df.to_pickle(gRNA_target_dir_path + f'{file_name}_target_df_{file_ind}.pkl')
                file_ind += 1
                file_path = valid_dir_path + f'{file_name}_valid_df_{file_ind}.pkl'

            report_df.loc['extract_target', file_name] = 1
            report_df.to_csv(config.fastq_path + 'report.csv')
        else:
            print(f'{file_name} has already been processed')

#########################################################################################
# 12 - Unite target dataframes
def unite_target_df(config, report_df):
    print('\n\n######################################')
    print('12. Unite target dataframes')
    unite_target_df_list = report_df.loc['unite_target_df', :]
    target_df_path = config.fastq_path + 'gRNA_target/'


    for file_name, val in unite_target_df_list.iteritems():
        if val == 0:
            print(f'Uniting {file_name} target dataframes files')
            file_index = 0
            unite_df = pd.DataFrame(columns=['read_gRNA', 'designed_gRNA_seq', 'prefix_range', 'suffix_ragne', 'read_prefix_seq',
                                             'read_gRNA_PAM', 'read_suffix_seq', 'is_edited'])

            while os.path.exists(target_df_path + file_name + f'_target_df_{file_index}.pkl'):
                file_path = target_df_path + file_name + f'_target_df_{file_index}.pkl'
                print(file_path)
                df_part = pd.read_pickle(file_path)
                df_part_valid = df_part[df_part.is_mis_synthesis == 0].drop(columns=['is_mis_synthesis'])
                unite_df = pd.concat([unite_df, df_part_valid], sort=False)
                file_index += 1

            unite_df.to_pickle(target_df_path + f'{file_name}_unified_df.pkl')

            report_df.loc['unite_target_df', file_name] = 1
            report_df.to_csv(config.fastq_path + 'report.csv')
        else:
            print(f'{file_name} has already been united')

#########################################################################################
# 13 - Calculate the final efficiency

def get_eff(df_gRNA, grp_dic_bg, grp_plasmid_edited):
    # First count the total number of reads
    absolute_counts = np.asscalar(df_gRNA.loc[:, 'read_counts'].sum())
    gRNA = df_gRNA.designed_gRNA_seq.values[0]
    if gRNA in grp_dic_bg.keys():
        # get the background database of the corresponding gRNA
        df_bg = grp_plasmid_edited.get_group(gRNA)
        # see wich ones are also presented in the bg library as edited in this read_barcode_target_seq
        cond1 = df_gRNA['read_gRNA_PAM'].isin(df_bg['read_gRNA_PAM'].values)
        # Which ones to exclude
        df_after_bg_correction = df_gRNA[~cond1]
    else:
        df_after_bg_correction = df_gRNA

    # df_after_bg_correction.to_csv(f'{gRNA}_ori.csv')
    reads_sum = np.asscalar(df_after_bg_correction.read_counts.sum()) if len(df_after_bg_correction.index) != 0 else 0
    df_edited_reads = df_after_bg_correction[df_after_bg_correction.is_edited == 1]

    edited_read_counts = np.asscalar(df_edited_reads.read_counts.sum()) if len(df_edited_reads.index) != 0 else 0
    non_edited_read_counts = reads_sum - edited_read_counts

    if reads_sum != 0:
        edit_efficiency = edited_read_counts / reads_sum
    else:
        edit_efficiency = np.NaN

    return gRNA, absolute_counts, reads_sum, edited_read_counts, non_edited_read_counts, edit_efficiency

def calc_efficiency(config, report_df):
    print('\n\n######################################')
    print('13. Calculate the final efficiency')
    calc_efficiency_list = report_df.loc['calc_efficiency', :]

    gRNA_target_dir_path = config.fastq_path + 'gRNA_target/'
    efficiency_dir_path = config.fastq_path + 'efficiency/'

    if not os.path.exists(efficiency_dir_path):
        os.mkdir(efficiency_dir_path)



    for rep in range(1, 3):

        for file_name, val in calc_efficiency_list.iteritems():
            if val == 0:
                if str(rep) in file_name:
                    print(f'Calc efficiency for {file_name}')
                    df_plasmid_edited = pd.read_pickle(gRNA_target_dir_path + f'background_{rep}_unified_df.pkl')
                    df_plasmid_edited = df_plasmid_edited[df_plasmid_edited.is_edited == 1]
                    grp_plasmid_edited = df_plasmid_edited.groupby('designed_gRNA_seq')
                    grp_dic_bg = {key: grp_plasmid_edited.get_group(key) for key in grp_plasmid_edited.groups.keys()}

                    df_valid_info = pd.read_pickle(gRNA_target_dir_path + file_name + '_unified_df.pkl')
                    df_valid_info['read_counts'] = df_valid_info.read_gRNA.apply(lambda x: np.int16(x.split('|')[1]))
                    grps = df_valid_info.groupby('designed_gRNA_seq')
                    grp_dic = {key: grps.get_group(key) for key in grps.groups.keys()}

                    data_list = []
                    for gRNA in list(grp_dic.keys()):
                        df_gRNA = grp_dic[gRNA]
                        r = get_eff(df_gRNA, grp_dic_bg, grp_plasmid_edited)
                        data_list.append(r)

                    df_efficiency = pd.DataFrame(data_list, columns=["gRNA","absolute_counts","reads_sum","edited_read_counts","non_edited_read_counts","edit_efficiency"])
                    df_efficiency.to_pickle(efficiency_dir_path + f'{file_name}_eff.pkl')


                    report_df.loc['calc_efficiency', file_name] = 1
                    report_df.to_csv(config.fastq_path + 'report.csv')
            else:
                print(f'{file_name} efficiency has already been calculated ')

#########################################################################################
# 14 - Write the final efficiencies to csv file

def get_mean_eff(row):
    read_sum = row['reads_sum_x'] + row['reads_sum_y']
    eff_x = 0 if row['reads_sum_x'] == 0 else row['reads_sum_x'] * row['edit_efficiency_x']
    eff_y = 0 if row['reads_sum_y'] == 0 else row['reads_sum_y'] * row['edit_efficiency_y']
    efficiency = (eff_x + eff_y) / read_sum
    edited_read_counts = row['edited_read_counts_x'] + row['edited_read_counts_y']
    return {'reads_sum': read_sum, 'edited_read_counts': edited_read_counts, 'mean_eff': efficiency}

def write_to_final_table(config, report_df):
    print('\n\n######################################')
    print('14. Writing efficiency to final csv for each enzyme separate')

    write_to_final_table_list = report_df.loc['write_to_final_table', :]
    efficiency_dir_path = config.fastq_path + 'efficiency/'


    for file_name, val in write_to_final_table_list.iteritems():
        enzyme = file_name.split('_')[0]
        if val == 0:
            print(f'Calculating the efficiency for {enzyme} enzyme')

            if enzyme == 'wt':
                final_eff = pd.read_pickle(efficiency_dir_path + f'wt_2_eff.pkl')
                final_eff.rename(columns = {'edit_efficiency': 'mean_eff'}, inplace = True)
                cond = final_eff.reads_sum >= 1
                final_eff = final_eff[cond]
                final_eff.to_csv(efficiency_dir_path + f'wt_final_eff.csv')
                report_df.loc['write_to_final_table', f'wt_2'] = 1
            else:
                eff_df_1 = pd.read_pickle(efficiency_dir_path + f'{enzyme}_1_eff.pkl')
                eff_df_2 = pd.read_pickle(efficiency_dir_path + f'{enzyme}_2_eff.pkl')
                final_eff = eff_df_1.merge(eff_df_2, on='gRNA')
                cond1 = final_eff.reads_sum_x >= 1
                cond2 = final_eff.reads_sum_y >= 1
                final_eff = final_eff[cond1 | cond2]
                final_eff_unified = final_eff.apply(lambda x: pd.Series(get_mean_eff(x)), axis=1)
                final_eff['reads_sum'] = final_eff_unified['reads_sum']
                final_eff['edited_read_counts'] = final_eff_unified['edited_read_counts']
                final_eff['mean_eff'] = final_eff_unified['mean_eff']

                final_eff.to_csv(efficiency_dir_path + f'{enzyme}_final_eff.csv')
                report_df.loc['write_to_final_table', f'{enzyme}_1'] = 1
                report_df.loc['write_to_final_table', f'{enzyme}_2'] = 1

            report_df.to_csv(config.fastq_path + 'report.csv')
        else:
            print(f'{enzyme} efficiency has already been calculated')



#########################################################################################
# 15 - Build final efficiency table without the bio-features.
def build_th_table(config):
    print('\n\n######################################')
    print(f'15. Building the full table without bio-features')
    efficiency_dir_path = config.fastq_path + 'efficiency/'
    if os.path.exists(efficiency_dir_path + 'final_efficiency.csv'):
        print('file has allready been created')
        return

    enzyme_list = ['wt', 'esp', 'hf']

    for enzyme in enzyme_list:
        print(f'Adding {enzyme} to the full table')
        enzyme_eff_df = pd.read_csv(efficiency_dir_path + f'{enzyme}_final_eff.csv')

        enzyme_eff_df = enzyme_eff_df.loc[:, ['gRNA', 'reads_sum', 'edited_read_counts', 'mean_eff']]
        enzyme_eff_df.columns = ['gRNA_Seq', f'{enzyme}_reads_sum', f'{enzyme}_edited_read_counts', f'{enzyme}_mean_eff']

        if enzyme == 'wt': # the first
            final_eff_df = enzyme_eff_df
        else:
            final_eff_df = pd.merge(final_eff_df, enzyme_eff_df, on='gRNA_Seq', how='outer')

    df_path = config.fastq_path + 'ref/df_ref_gRNA_choosen.pkl'
    df_ref_gRNA_choosen = pd.read_pickle(df_path)
    gRNA_PAM_df = df_ref_gRNA_choosen.loc[:, ['target_gRNA_PAM']]
    gRNA_PAM_df.rename(columns={'target_gRNA_PAM': '21mer'}, inplace=True)
    gRNA_PAM_df.sort_values(by=['21mer'], inplace=True)
    gRNA_PAM_df.reset_index(inplace=True, drop=True)

    gRNAs_line = gRNA_PAM_df.apply(lambda x: x['21mer'][:20], axis=1)
    mer = gRNA_PAM_df.apply(lambda x: x['21mer'][:21], axis=1)
    gRNA_index = gRNA_PAM_df.apply(lambda x: f'gRNA-{x.name}', axis=1)

    gRNA_PAM_df['gRNA'] = gRNA_index
    gRNA_PAM_df['21mer'] = mer
    gRNA_PAM_df['gRNA_Seq'] = gRNAs_line
    gRNA_PAM_df = gRNA_PAM_df[['gRNA', '21mer', 'gRNA_Seq']]

    final_eff_df = pd.merge(gRNA_PAM_df, final_eff_df, on="gRNA_Seq", how="left")
    final_eff_df.to_csv(efficiency_dir_path + 'final_efficiency.csv', index=False)


#########################################################################################
# 18 - Adding the bio-features to the final table.


def add_biofeatures(config):
    print('\n\n######################################')
    print(f'15. Building the full table without bio-features')
    efficiency_dir_path = config.fastq_path + 'efficiency/'

    final_efficiency = pd.read_csv(efficiency_dir_path + 'final_efficiency.csv')

    feature_options = {
        "testing_non_binary_target_name": 'ranks',
        'include_pi_nuc_feat': True,
        "gc_features": True,
        "nuc_features": True,
        "include_Tm": True,
        "include_structure_features": True,
        "order": 3,
        "num_proc": 20,
        "normalize_features": None
    }
    feature_sets = feature_util.featurize_data(final_efficiency, feature_options)
    # feature_sets['dG_features'].reset_index(inplace=True)
    # dG_features = feature_sets['dG_features']
    # dG_features.reset_index(inplace=True)
    # final_efficiency = pd.concat([final_efficiency, dG_features], axis=1)

    for feature in feature_sets.keys():
        print(feature)
        reindexed_feature_df = feature_sets[feature]
        reindexed_feature_df.reset_index(inplace=True, drop=True)
        final_efficiency = pd.concat([final_efficiency, reindexed_feature_df], axis=1)
    # gc_above_10 = feature_sets['gc_above_10']
    # gc_below_10 = feature_sets['gc_below_10']
    # gc_count = feature_sets['gc_count']
    # Tm = feature_sets['Tm']
    # dG_features = feature_sets['dG_features']
    # dG_features.reset_index(inplace=True)
    #
    # final_efficiency = pd.concat([final_efficiency, dG_features, gc_above_10, gc_below_10, gc_count, Tm], axis=1)



    final_efficiency.to_csv(efficiency_dir_path + 'final_efficiency_with_bio.csv', index=False)


# Main function
def create_data(config):
    print('Preprocessing fastq files')
    report_df = create_csv(config)
    split_files(config, report_df)
    drop_N(config, report_df)
    fq2df(config, report_df)
    drop_scaffold(config, report_df)
    unite_dataframes(config, report_df)
    create_fasta(config, report_df)
    df_ref_gRNA_choosen = analyze_ref(config)  # currently is used manually with bwa in cb1 - TODO - run in server with script
    align_to_genome(config, report_df)
    split_valid_df(config, report_df)
    extract_targets(config, report_df)
    unite_target_df(config, report_df)
    calc_efficiency(config, report_df)
    write_to_final_table(config, report_df)
    build_th_table(config)
    add_biofeatures(config)

