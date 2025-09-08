import argparse,joblib,yaml
from  pathlib import Path
#from data_analysis_tk.analysis_task import task_save#, load_map
#from data_analysis_tk.analysis_task import read_task#,load_df_rmap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict,Counter

import positions


'''
this script is meant to 
- make analysis streamlined
- self document the process
- a form of howto

will use data_batches (ascii files) that instruct what to do with the data sets.

'''
def load_map(mapfile):
    d = {}
    if not mapfile.endswith('.map'):
        mapfile = f'{mapfile}.map'
    with open(mapfile,'r') as f:
        for line in [x.strip() for x in f.readlines() if x.strip()]:
            if line=='EOF':
                break
            #print('line ', line)
            k,v = [y.strip() for y in line.split(':')]
            d[k] = v
    return d

def load_df_rmap(taskn,ws=None):
    meta = read_task(taskn)
    duf = pd.read_csv(meta['behave_csv'])
    rmap = load_map(meta['rename_map'])
    return duf,rmap
def read_task(taskname):
    """
    Reads a .task YAML file and returns a dict of metadata.
    Raises FileNotFoundError if not found.
    """
    meta = {}
    meta_path = str((Path('.') / f'data/{taskname}.task').resolve())
    if not Path(meta_path).exists():
        raise FileNotFoundError(f"[read_task] Metadata file not found: {meta_path}")
    else:
        with open(Path(meta_path), "r") as f:
            meta = yaml.safe_load(f)
    return meta

def task_save(metaname,meta):
    if 'meta_path' in meta:
        save_path = f'{meta["meta_path"]}/{metaname}.task'
    else:
        save_path = str((Path('.') / f'data/{metaname}.task').resolve())
        meta["meta_path"] = str(Path('.').resolve())
    with open(save_path, "w") as f:
        yaml.dump(meta, f, sort_keys=False)
    print(f"Metadata written to {save_path}")


def load_samples(clusterfile):
    combined_dfs = []
    start_id = 0
    lines=[]
    sz=0
    with open(clusterfile,'r') as fp:
        lines = [x.strip() for x in fp.readlines() if x.strip()]
    for line in lines:
        args = [x.strip() for x in line.split()]
        task  = args[0]
        sz = int(args[1])
        fields = args[2:]
        rqa_dict = joblib.load(f'/one/work_caches/rqa_dfs/{task}_{sz}_{sz}_rqa_df.joblib')
        meta = read_task(task)
        rename_map = load_map(meta['rename_map'])
        for field in fields:
            field = rename_map[field]
            df = rqa_dict[field].copy()
            #df = df.copy()
            df["trace_id"] = f'{task}_{sz}'
            df["trace_id_field"] = f"{task}_{sz}_{field}"
            df["original_window_id"] = df["window_id"]
            df["field"] = field
            n = len(df)
            df["global_window_id"] = range(start_id, start_id + n)
            start_id += n
            combined_dfs.append(df)
    rqa_all_df = pd.concat(combined_dfs, ignore_index=True)
    return rqa_all_df,sz


def cluster_fit_and_predict(rqa_all_df,features_cols=None,n_clusters=8,window_size=128
                            ,filename='neben'):
    features_cols = features_cols or ["RR", "DET", "LAM", "TT", "RPDE", "ENTR", "DIV", "causal_bias"]
    features = rqa_all_df[features_cols].fillna(0)  # Fill NaN for clustering

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    n_clusters = n_clusters or 6  # Or try silhouette analysis

    algo = 'kmeans'
    if algo=='kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42)
        model.fit(X_scaled)
        rqa_all_df["cluster_id"] = model.predict(X_scaled)

    symbolic_traces = {
        trace_id_field: group.sort_values("global_window_id")["cluster_id"].tolist()
        for trace_id_field, group in rqa_all_df.groupby("trace_id_field")
    }

    joblib.dump((rqa_all_df, symbolic_traces,n_clusters,window_size,scaler,features_cols,model)
                , f"data/clustered_rqa_and_symbolic_traces_{filename[:-4]}.joblib")
    print(f'{filename} clustering dumped alright')


def cluster_predict(file,cluster_model):
    rqa_all_df,window_size = load_samples(file)
    symboloc = f"data/clustered_rqa_and_symbolic_traces_{cluster_model[:-4]}.joblib"
    _,_,n_clusters,window_size,scaler,features_cols,model = joblib.load(symboloc)

    features = rqa_all_df[features_cols].fillna(0) 
    X_scaled = scaler.transform(features)
    rqa_all_df["cluster_id"] = model.predict(X_scaled)
    symbolic_traces = {
        trace_id_field: group.sort_values("global_window_id")["cluster_id"].tolist()
        for trace_id_field, group in rqa_all_df.groupby("trace_id_field")
    }
    joblib.dump((rqa_all_df, symbolic_traces,n_clusters,window_size,scaler,features_cols,model)
                , f"data/clustered_rqa_and_symbolic_traces_{file[:-4]}.joblib")
    print(f'{file} clustering dumped alright')




def build_behavior_signatures(rqa_df, field, context_dict, semantic_field="_source.rule.mitre.tactics"):
    signatures = []

    for _, row in rqa_df.iterrows():
        trace_id = row.get("trace_id")
        win_id = int(row["original_window_id"])
        key = (trace_id, win_id)

        # Cluster label
        cluster = row["cluster_id"]

        # Semantic: extract most common tactic
        semantic = "Unknown"

        if key in context_dict and semantic_field in context_dict[key]:
            semantic_counts = context_dict[key][semantic_field]
            if semantic_counts and isinstance(semantic_counts, Counter):
                semantic = semantic_counts.most_common(1)[0][0]
            elif isinstance(semantic_counts, list):
                if semantic_counts:
                    semantic = semantic_counts[0]

        # RQA metrics
        RR = float(row.get("RR", 0))
        LAM = float(row.get("LAM", 0))
        RPDE = float(row.get("RPDE", 0))
        DET = float(row.get("DET", 0))
        DIV = float(row.get("DIV", 0))

        # Scoring
        intensity = (RR + LAM + RPDE) / 3
        focus = (DET + (1 - DIV)) / 2

        signatures.append({
            "trace_id": trace_id,
            "window_id": win_id,
            "cluster": cluster,
            "semantic": semantic,
            "intensity": intensity,
            "focus": focus
        })

    return signatures



def fingerprint_updated(file_batch,sseq=None):
    symboloc = f"data/clustered_rqa_and_symbolic_traces_{file_batch[:-4]}.joblib"
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

        # for test speed up
        #if r[0].startswith('alf'):
            #continue

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
                
        pat = f'data/context_dicts/{task}_{size}_cnxmap.joblib'
        context_dict=joblib.load(pat)

        rqa_df = rqa_df.copy().sort_values(by=["trace_id", "original_window_id"])
        signatures = build_behavior_signatures(rqa_df, 'execve.a0'
                            ,context_dict
                            , semantic_field= rmap[pfield])
        internal_dict['signatures'] = signatures

        internal_dict['symbolic_trace'] = symbolic_traces[trace_id_field]


        seqs={}
        internal_dict['cs']  = seqs


        #### trace fingerprinting
        rg = lambda m,row:float(row.get(m,0))
        met = lambda row : int((rg("RR",row)+rg("LAM",row)+rg("RPDE",row)/4.) // .25)

        sequence = [(row['cluster_id'],met(row)) for _,row in rqa_df.iterrows()]



        print(f'trace_id {trace_id_field}')
        cb = positions.reconstruct_eam(sequence)
        #for p in cb:
            #print(set(p))


    joblib.dump(fingerprints,f"data/fingerprints_{file_batch[:-4]}.joblib")
    print(f'{file_batch} fingerprints dumped')





def main():
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument("mode" , nargs=1, default=False)
    parser.add_argument("args",nargs="+", default=False)
    parser.add_argument("--model", nargs="?", default=False)
    args = parser.parse_args()

    if args.mode[0] in ['model']:
        # cluster fit and predict.
        rqa_all_df,window_size = load_samples(args.args[0]) #file_batch
        cluster_fit_and_predict(rqa_all_df,n_clusters=args.n_clusters
                                ,window_size=window_size,filename=args.args[0])
        print(f'[modeling] Clustering Done')
        fingerprint(args.args[0],args.sseq)
        print(f'[modeling] Fingerprinting Done')
        pooling(args.args[0])
        print(f'[modeling] Pooling Done')

    elif args.mode[0] in ['eam']:
        for file in args.args:
            cluster_predict(file,args.model)
            fingerprint_updated(file,False)

if __name__ == '__main__':
    main()
