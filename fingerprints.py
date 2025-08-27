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
                
        pat = f'/one/work_caches/context_dicts/{task}_{size}_cnxmap.joblib'
        context_dict=joblib.load(pat)

        rqa_df = rqa_df.copy().sort_values(by=["trace_id", "original_window_id"])
        signatures = build_behavior_signatures(rqa_df, 'execve.a0'
                            ,context_dict
                            , semantic_field= rmap[pfield])
        internal_dict['signatures'] = signatures

        internal_dict['symbolic_trace'] = symbolic_traces[trace_id_field]


        seqs={}
        if sseq:
            semantic_field = sseq # 'command'
            # Compute TF-IDF semantics (only once)
            tfidf_semantics = get_tfidf_representative_terms(rqa_df
                                 ,context_dict , semantic_field=rmap[semantic_field],top_k=3)

            # Build the sequence
            seqs = build_cluster_semantic_sequences_with_tfidf(rqa_df, tfidf_semantics)

            #sequences = build_cluster_semantic_sequences(rqa_df, pfield
                                         #, context_dict , semantic_field=rmap[semantic_field])
        
        internal_dict['cs']  = seqs


        #### trace fingerprinting
        rg = lambda m,row:float(row.get(m,0))
        met = lambda row : int((rg("RR",row)+rg("LAM",row)+rg("RPDE",row)/4.) // .25)

        sequence = [(row['cluster_id'],met(row)) for _,row in rqa_df.iterrows()]
        motifs = extract_motifs_collapsed(sequence, n=3)
        #print('sequence ', sequence)
        #print('motifs ', motifs)

        motifdistribution = motif_distribution(motifs)
        #print('motifdistribution ', motifdistribution)

        motifs_for_markov = extract_motifs_collapsed(sequence, n=2)
        motifmarkov = motif_transition_fingerprints(motifs_for_markov)



        print(f'trace_id {trace_id_field}')
        cb = positions.reconstruct_eam(sequence)
        #for p in cb:
            #print(set(p))


        #follows, precedes = compute_order_relation_matrices(sequence,win_size=None)
        follows, precedes = gof_compute_order_relation_matrices(sequence)
        #follows, precedes = compute_order_relation_matrices(sequence)
        #follows,precedes,ifollows,iprecedes = bb_cworder_rel(sequence)
        follows,precedes,ifollows,iprecedes = bb_compute_order_relation_matrices(sequence,win_size=64)


        if 0:
            if (np.array(follows.values) == np.array(precedes.values)).all():
                print(f'[[[[[[[[[[[[[[{trace_id_field}, follows == precedes]]]]]]]]]]]]]]')
                print(follows)
            else:
                print(f'[------------{trace_id_field} fingered -------------]')

        global_states = None#list(follows.index)

        result = compute_rolling_never_matrices(sequence,win_size=32,global_states=global_states)
        nf = compute_rolling_never_matrices_j(sequence,32,32)

        internal_dict['cm'] = \
                (sequence,motifs,motifdistribution,motifmarkov,follows,precedes,result,nf)

        fingerprints[trace_id_field] = internal_dict

        if 0:
            if trace_id_field.startswith('horf'):
                print('horf check')
                print('sequence ', sequence)


    joblib.dump(fingerprints,f"fingerprints_{file_batch[:-4]}.joblib")
    print(f'{file_batch} fingerprints dumped alright')




def main():
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument("file_batch",nargs="+")
    #parser.add_argument("symbolic_traces_file")
    parser.add_argument("--cs", nargs="?", default=False)
    parser.add_argument("--sseq", nargs="?", default=False)
    parser.add_argument("--cm", nargs="?", default=False)
    parser.add_argument("--st", action="store_true")
    parser.add_argument("--cluster_semantics", action="store_true")
    parser.add_argument("--motif", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plott", action="store_true")
    parser.add_argument("--markov", action="store_true")
    parser.add_argument("--entropy", action="store_true")
    parser.add_argument("--order", action="store_true")
    args = parser.parse_args()

    #print('args.file_batch ', args.file_batch)

    for file in args.file_batch:
        symboloc = f"clustered_rqa_and_symbolic_traces_{file[:-4]}.joblib"
        rqa_all_df,symbolic_traces,n_clusters,window_size,scaler,features_cols,model = \
                joblib.load(symboloc)

        rqa_by_trace = {
            trace_id: group.copy()
            for trace_id, group in rqa_all_df.groupby("trace_id_field")
        }


        fingerprints = {}
        for trace_id_field,rqa_df in rqa_by_trace.items():
            internal_dict = {}
            #print('trace_id_field ', trace_id_field)
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
                    
            pat = f'/one/work_caches/context_dicts/{task}_{size}_cnxmap.joblib'
            context_dict=joblib.load(pat)

            rqa_df = rqa_df.copy().sort_values(by=["trace_id", "original_window_id"])
            signatures = build_behavior_signatures(rqa_df, 'execve.a0'
                                ,context_dict
                                , semantic_field= rmap[pfield])
            internal_dict['signatures'] = signatures

            internal_dict['symbolic_trace'] = symbolic_traces[trace_id_field]


            seqs={}
            if args.sseq:
                semantic_field = sseq # 'command'
                # Compute TF-IDF semantics (only once)
                tfidf_semantics = get_tfidf_representative_terms(rqa_df
                                     ,context_dict , semantic_field=rmap[semantic_field],top_k=3)

                # Build the sequence
                seqs = build_cluster_semantic_sequences_with_tfidf(rqa_df, tfidf_semantics)

                #sequences = build_cluster_semantic_sequences(rqa_df, pfield
                                             #, context_dict , semantic_field=rmap[semantic_field])
            
            internal_dict['cs']  = seqs


            #### trace fingerprinting
            rg = lambda m,row:float(row.get(m,0))
            met = lambda row : int((rg("RR",row)+rg("LAM",row)+rg("RPDE",row)/4.) // .25)

            sequence = [(row['cluster_id'],met(row)) for _,row in rqa_df.iterrows()]
            motifs = extract_motifs_collapsed(sequence, n=2)
            #print('sequence ', sequence)
            #print('motifs ', motifs)

            motifdistribution = motif_distribution(motifs)
            #print('motifdistribution ', motifdistribution)
            motifmarkov = motif_transition_fingerprints(motifs)
            #follows, precedes = compute_order_relation_matrices(sequence,win_size=32)
            follows, precedes = compute_order_relation_matrices(sequence)

            #if (np.array(follows.values) == np.array(precedes.values)).all():
                #print(trace_id_field)
                #1/0
            global_states = None#list(follows.index)

            result = compute_rolling_never_matrices(sequence,win_size=32,global_states=global_states)
            nf = compute_rolling_never_matrices_j(sequence,32,32)

            internal_dict['cm'] = \
                    (sequence,motifs,motifdistribution,motifmarkov,follows,precedes,result,nf)

            fingerprints[trace_id_field] = internal_dict
            print(trace_id_field,'follows')


        joblib.dump(fingerprints,f"fingerprints_{file[:-4]}.joblib")
        print(f'{file} fingerprints dumped alright')



if 0:
    print('dist2 ', dist2)

    #r = compute_entropy_measures(sequence)
    #print('r ', r)
    follows, precedes = compute_order_relation_matrices(sequence)
    md={}
    #print(field)
    print("=== FOLLOWS MATRIX ===")
    print(follows)
    md['follows'] = follows

    print("\n=== PRECEDES MATRIX ===")
    print(precedes)
    md['precedes'] = precedes



if __name__ == "__main__":
    main()
