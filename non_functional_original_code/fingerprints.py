import argparse,joblib
from data_analysis_tk.analysis_task import read_task,task_save, load_map,load_df_rmap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict,Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from symbol_sequence_analysis import extract_motifs_collapsed\
        ,gof_compute_order_relation_matrices,compute_entropy_measures,compute_rolling_never_matrices\
        ,compute_rolling_never_matrices_j,bb_order_rel,bb_compute_order_relation_matrices
from semantic_analysis import build_behavior_signatures,get_tfidf_representative_terms\
        ,build_cluster_semantic_sequences_with_tfidf


import positions

#### trace fingerprinting
'''
fingerprinting does same things that semantic_analysis does
just that it does not provide options and acts on the whole pkg from cluster_rqa
'''


def motif_distribution(motifs, all_motifs=None):
    total = sum(motifs.values())
    if total == 0:
        return {}

    dist = {m: count / total for m, count in motifs.items()}

    # Optionally align with global motif set
    if all_motifs:
        dist = {m: dist.get(m, 0.0) for m in all_motifs}
    
    return dist


def motif_transition_fingerprints(motif_counter):
    """
    Build transition distributions conditioned on current motif state.
    Returns:
        - prob_dict: dict of dicts {curr_state: {next_state: prob, ...}}
        - df: pandas DataFrame [curr_state x next_state] of probabilities
    """
    transition_dict = defaultdict(Counter)

    # Accumulate counts
    for (curr, nxt), count in motif_counter.items():
        transition_dict[curr][nxt] += count

    # All possible states (from both current and next)
    all_curr = set(transition_dict.keys())
    all_next = {nxt for nexts in transition_dict.values() for nxt in nexts}
    all_states = sorted(all_curr.union(all_next))

    # Create DataFrame with all combinations
    df = pd.DataFrame(index=all_states, columns=all_states, data=0.0)

    for curr in transition_dict:
        row_total = sum(transition_dict[curr].values())
        for nxt in transition_dict[curr]:
            if row_total > 0:
                df.at[curr, nxt] = transition_dict[curr][nxt] / row_total

    # Fill NaNs with 0 (if any)
    df = df.fillna(0.0)

    # Optional dict view
    prob_dict = {
        curr: {nxt: df.at[curr, nxt] for nxt in all_states if df.at[curr, nxt] > 0}
        for curr in all_states
    }

    return prob_dict, df


def fingerprint(file_batch,sseq=None):
    symboloc = f"clustered_rqa_and_symbolic_traces_{file_batch[:-4]}.joblib"
    print('symboloc ', symboloc)
    rqa_all_df,symbolic_traces,n_clusters,window_size,scaler,features_cols,model = \
            joblib.load(symboloc)

    rqa_by_trace = {
        trace_id: group.copy()
        for trace_id, group in rqa_all_df.groupby("trace_id_field")
    }


    fingerprints = {}
    for trace_id_field,rqa_df in rqa_by_trace.items():
        internal_dict = {}
        print('trace_id_field ', trace_id_field)
        r = trace_id_field.split('_')
        if len(r)==4:
            task,size,_,field = r
        elif len(r) == 3:
            task,size,field = r
        full_df,rmap = load_df_rmap(task,size)
        #print('rmap ', rmap)

        for x,y in rmap.items():
            pfield = x
            if y.startswith('_') and y[1:]==field:
                field = f'_{field}'
                break
                
        pat = 'path-to-joblib-file'
        context_dict=joblib.load(pat)

        rqa_df = rqa_df.copy().sort_values(by=["trace_id", "original_window_id"])
        signatures = build_behavior_signatures(rqa_df, 'execve.a0'
                            ,context_dict
                            , semantic_field= rmap[pfield])
        internal_dict['signatures'] = signatures

        internal_dict['symbolic_trace'] = symbolic_traces[trace_id_field]



        #### trace fingerprinting
        rg = lambda m,row:float(row.get(m,0))
        met = lambda row : int((rg("RR",row)+rg("LAM",row)+rg("RPDE",row)/4.) // .25)

        sequence = [(row['cluster_id'],met(row)) for _,row in rqa_df.iterrows()]


        print(f'trace_id {trace_id_field}')
        cb = positions.reconstruct_eam(sequence)
        #for p in cb:
            #print(set(p))



    joblib.dump(fingerprints,f"fingerprints_{file_batch[:-4]}.joblib")
    print(f'{file_batch} fingerprints dumped alright')




def main():
        pass

if __name__ == "__main__":
    main()
