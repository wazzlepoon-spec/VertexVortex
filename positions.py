import itertools
import re
import bisect
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from collections import defaultdict, Counter, deque

'''
logic fragment
program
extraction
consistency
'''

class Atom:
    def __init__(self, name, negated=False):
        m = re.match('([~,-]*)\s*([a-z)(]+[0-9)]*)',name)
        if m:
            neg,atom = m.groups()
        self.name = atom  # string
        self.negated = False if len(neg)%2==0 else True
        if negated:
            self.negated = not self.negated

    def negate(self):
        return Atom(self.name, not self.negated)

    def apply(self, func_name):
        # Just string-level application: f(x) becomes "f(x)"
        return Atom(f"{func_name}({self.name})", self.negated)

    def precond(self):
        return self.apply('p')

    def __str__(self):
        return f"~{self.name}" if self.negated else self.name

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, Atom) and self.name == other.name and self.negated == other.negated

    def __hash__(self):
        return hash((self.name, self.negated))

class ModalFormula:
    def __init__(self, *atoms,op=None):
        self.atoms = frozenset([Atom(x) if isinstance(x,str) else x  for x in atoms])
        self.atom = Atom(atoms[0]) if isinstance(atoms[0],str) else atoms[0] if len(atoms)==1 else None
        self.op = op

    def conj_str(self):
        if not self.atoms:
            return "⊤"  # True, empty conjunction
        return f"({'.'.join(str(a) for a in self.atoms)})" if len(self.atoms)>1 else f"{str(self.atom)}"
    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            set(self.atoms) == set(other.atoms) and
            self.op == other.op
        )
    def __hash__(self):
        return hash((self.__class__, frozenset(self.atoms),self.op))



class Box(ModalFormula):
    def __str__(self):
        return f"{self.op if self.op else '□'}{self.conj_str()}"
    def __repr__(self):
        return str(self)


class NegBox(ModalFormula):
    def __str__(self):
        #return f"~□{self.conj_str()}"
        return f"~{self.op if self.op else '□'}{self.conj_str()}"
    def __repr__(self):
        return str(self)

class BoxNeg(ModalFormula):
    def __str__(self):
        #return f"□~{self.conj_str()}"
        return f"{self.op if self.op else '□'}~{self.conj_str()}"
    def __repr__(self):
        return str(self)


class NegBoxNeg(ModalFormula):
    def __str__(self):
        #return f"~□~{self.conj_str()}"
        return f"~{self.op if self.op else '□'}~{self.conj_str()}"
    def __repr__(self):
        return str(self)


class Conjunction:
    def __init__(self, *formulas):
        self.formulas = list(formulas)
    def __str__(self):
        return " ∧ ".join(str(f) for f in self.formulas)
    def __repr__(self):
        return str(self)
    def __eq__(self, other):
        return isinstance(other, Conjunction) and set(self.formulas) == set(other.formulas)
    def __hash__(self):
        return hash(frozenset(self.formulas))

class Variable:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name
    def __repr__(self):
        return str(self)
    def __eq__(self, other):
        return isinstance(other, Variable) and self.name == other.name
    def __hash__(self):
        return hash(self.name)

def apply_substitution(f, subst):
    atoms = [subst.get(a, a) if isinstance(a, Variable) else a for a in f.atoms]
    return type(f)(*atoms)

def extract_atoms(formula):
    return list(formula.atoms)

def extract_vars(formula):
    return [a for a in formula.atoms if isinstance(a, Variable)]

def form(rule,*observables):
    pass
def form(rule,ctl,a,b):
    pass


def general_modal_rule(s):
    return [
        Box(s),
        Box(tick(s)),
    ]

#2. programs =================


def index_positions(seq):
    pos = defaultdict(list)
    for i, x in enumerate(seq):
        pos[x].append(i)
    return dict(pos)

def between_sets_first_b_after_a(seq, a, b):
    """
    For each occurrence of 'a' in seq, find the first 'b' after it.
    Return a list of sets: symbols strictly between that 'a' and that 'b'.
    If there is no later 'b' for an 'a', that 'a' is skipped.
    """
    pos = index_positions(seq)
    if a not in pos or b not in pos:
        return []

    res = []
    b_pos = pos[b]                      # sorted list of indices of b
    for i in pos[a]:                    # each index of a
        k = bisect.bisect_right(b_pos, i)
        if k == len(b_pos):
            continue                    # no b after this a → skip
        j = b_pos[k]                    # first b after this a
        res.append(set(seq[i+1:j]))     # symbols between
    return res
def between_sets_last_a_before_b(seq, a, b):
    """
    Scan seq; whenever we see 'a', start a new window (reset).
    Collect symbols until we see 'b'; then emit that set and close the window.
    If another 'a' appears before 'b', reset (discard what was collected) and start over.
    Returns: list of sets, one per (last-a ... first-b) window.
    """
    out = []
    collecting = False
    cur = set()
    for x in seq:
        if x == a:
            # start/restart window at this a
            collecting = True
            cur.clear()
        elif collecting:
            if x == b:
                out.append(set(cur))  # snapshot
                collecting = False
                cur.clear()
            elif x != a:              # never include 'a' in the set
                cur.add(x)
    return out

def between_union_last_a_before_b(seq, a, b):
    u = set()
    for s in between_sets_last_a_before_b(seq, a, b):
        u |= s
    return u


def preprocess_between(seq):
    """Prepare fast 'is there a symbol c with a < c < b appearing in seq?' queries."""
    uniq_sorted = sorted(set(seq))             # total order of seen symbols
    present = [1]*len(uniq_sorted)             # since they're all present by construction
    pref = [0]
    for v in present:
        pref.append(pref[-1] + v)              # prefix sums over ranks
    idx = {s:i for i,s in enumerate(uniq_sorted)}
    return uniq_sorted, pref, idx

def exists_between(a, b, prep):
    """Return True iff ∃ c in seq with a < c < b (value-wise), given a<b."""
    uniq_sorted, pref, idx = prep
    # if a or b weren’t in the sequence, still okay: use their would-be ranks
    ra = bisect.bisect_left(uniq_sorted, a)
    rb = bisect.bisect_left(uniq_sorted, b)
    
    if not (a in idx and b in idx):            # optional: require both seen
        pass
    if ra >= rb:                               # covers a>=b or b≤a-in-order
        return False
    # count of symbols with ranks in (ra, rb) intersected with those present
    # present==1 for all uniq_sorted items, so this is just: (rb - (ra+1))>0
    return (pref[rb] - pref[ra+1]) > 0

# demo
seq = list("abcdcb")
prep = preprocess_between(seq)
print(exists_between('a','d', prep))  # True (b or c exist)
print(exists_between('c','d', prep))  # False (no symbol strictly between c and d)
print(exists_between('c','f', prep))  # True if any of {d,e} is in seq (here: d)


def reach_after_before(seq, hops=0,allow_self=False, dtype=int):
    """
    seq: list/array/Series of symbols (strings/chars/etc.)
    F[i,j] = 1  iff some j occurs strictly after some i
    P = F.T
    allow_self=False -> F[i,i]=1 only if there's a later i (repeat)
    """
    # Alphabet in stable order of first appearance (any order is fine)
    syms = pd.Index(pd.unique(pd.Series(seq)))
    idx = {s: k for k, s in enumerate(syms)}
    n = len(syms)

    F = np.zeros((n, n), dtype=bool)      # after
    seen = np.zeros(n, dtype=bool)        # symbols seen in the suffix

    # scan right->left
    for x in reversed(seq):
        u = idx[x]
        # all symbols seen to the right occur after x
        if seen.any():
            F[u, :] |= seen
        # if you *don’t* want self edges unless there's a repeated x later:
        # seen[u] being False here ensures we won’t set F[u,u] yet.
        seen[u] = True

    if not allow_self:
        np.fill_diagonal(F, False)

    F_df = pd.DataFrame(F.astype(dtype), index=syms, columns=syms)
    P_df = F_df.T
    return F_df, P_df


def reach_bounded_hops_counts(seq, H, allow_self=False):
    """
    Count ordered pairs (i,j) where j occurs within <= H positions after i.
    Returns (F_count, F_bool) as DataFrames (rows: i, cols: j).
    """
    syms = list(dict.fromkeys(seq))             # stable unique order
    pos  = {s:i for i,s in enumerate(syms)}
    nA   = len(syms)
    C    = np.zeros((nA, nA), dtype=np.int64)

    window = deque()        # last H symbols’ indices
    counts = Counter()      # multiset of symbols in window

    for x in seq:
        j = pos[x]
        # all symbols currently in window precede this j by <= H
        for i_sym, cnt in counts.items():
            if allow_self or i_sym != j:
                C[i_sym, j] += cnt
        # push current symbol, pop left if window > H
        window.append(j)
        counts[j] += 1
        if len(window) > H:
            old = window.popleft()
            counts[old] -= 1
            if counts[old] == 0:
                del counts[old]

    F_count = pd.DataFrame(C, index=syms, columns=syms)
    if not allow_self:
        np.fill_diagonal(F_count.values, 0)
    F_bool = (F_count > 0).astype(int)
    return F_count, F_bool

def reach_after_before_counts(self,seq):
    """
    seq: iterable of symbols (str/int/etc.)
    Returns:
      F_bool[i,j] = 1 iff ∃ positions t<u with seq[t]=i, seq[u]=j
      P_bool      = F_bool.T
      F_count[i,j]= # of ordered pairs (t,u), t<u, with seq[t]=i, seq[u]=j
    """
    syms = pd.Index(pd.unique(pd.Series(seq)))  # stable-order alphabet
    pos = {s:k for k,s in enumerate(syms)}
    n = len(syms)

    # frequency of symbols seen so far (left->right)
    seen = np.zeros(n, dtype=np.int64)
    C = np.zeros((n, n), dtype=np.int64)  # counts of pairs (i before j)

    for x in seq:
        j = pos[x]
        # all previously seen i contribute pairs (i -> current j)
        C[:, j] += seen
        # now include current j in the history
        seen[j] += 1

    F_bool = (C > 0).astype(int)
    P_bool = F_bool.T
    F_count = pd.DataFrame(C, index=syms, columns=syms)
    P_count = F_count.T
    return pd.DataFrame(F_bool, index=syms, columns=syms),\
            pd.DataFrame(P_bool, index=syms, columns=syms),\
            F_count,P_count


def prune_min_count(F_count, min_count=2):
    M = F_count.copy()
    M.values[M.values < min_count] = 0
    return M

def topk_counts(F_count, k):
    M = F_count.values.copy()
    keep = np.zeros_like(M, dtype=bool)
    for r in range(M.shape[0]):
        row = M[r]
        if np.count_nonzero(row) <= k:
            keep[r] = row > 0
        else:
            idx = np.argpartition(-row, kth=k)[:k]
            keep[r, idx] = True
    M[~keep] = 0
    return pd.DataFrame(M, index=F_count.index, columns=F_count.columns)
class Program:
    def __init__(self,program_order,dic):
        self.dic = dic
        self.program_order = program_order
        self.sequence_order = program_order[::-1]
        if 0:
            self.oldF,self.oldP = self.make_FP()
            self.newF = self.build_adj_from_seq(self.sequence_order)
            self.newP = self.newF.T
            u = self.build_adj_from_seq(self.sequence_order)
            print('u ', u)
            self.tcF,self.tcP = self.transitive_closure(self.newF)
            #f,p = self.reach_after_before(self.sequence_order)
            self.rabF,self.rabP,self.rabFC,self.rabPC = self.reach_after_before_counts(self.sequence_order)
            s = self.hop_stats(self.sequence_order)
            print(s)

    def succ(self,a):
        if a!=self.program_order[-1]:
            return self.program_order[self.program_order.index(a)+1]
    def prec(self,a):
        if a!=self.program_order[0]:
            return self.program_order[self.program_order.index(a)-1]
    def g(self,a):
        return self.dic[a][0]
    def u(self,a):
        return self.dic[a][1]
    def printp(self):
        for a in self.program_order:
            print(f'{self.dic[a][0]}\tafter\t{a}\tgiven\t{self.dic[a][1]}')

    def check_eq(self,df1,df2):
        for x in df1.index:
            for y in df1.columns:
                if df1.loc[x,y]>0:
                    if df2.loc[x,y]==0:
                        print('not equal')
                        return
        print('equal')

    def extract_belief(self,test_sequence,token,next):
        fwd,pre = self.make_FP(test_sequence)
        print('test_sequence ', test_sequence)
        #F,P = self.make_FP(self.sequence_order)
        print(fwd)
        aggregator = []
        belief_state = []
        if 0:
            for x in fwd.index:
                for y in fwd.columns:
                    f = fwd.loc[x,y]
                    F = self.F.loc[x,y]
                    p = pre.loc[x,y]
                    P = self.P.loc[x,y]
        for i,token in enumerate(test_sequence):
            next = test_sequence[i+1]

    def new_dong(self,test_sequence,token):
        #new=[Box(token)]
        new=[]
        appliedrule=None
        #fwd,pre = self.make_FP(test_sequence)
        fwd = self.build_adj_from_seq(test_sequence)
        pre = fwd.T 
        #print(pre)

        df = self.newF.copy()
        cols_with_1 = df.columns[df.loc[token] > 0].tolist()
        cols_with_0 = df.columns[df.loc[token] == 0].tolist()
        #print('cols_with_1 ', cols_with_1)
        #print('cols_with_0 ', cols_with_0)
        #print('fwd ', fwd)
        for c in cols_with_1:
            if c in fwd.columns:
                #print('token,c ', token,c)
                f = fwd.loc[token,c]
                if f:
                    new.extend([NegBoxNeg(token),Box(token)])
                else:
                    new.extend([NegBoxNeg(token),NegBox(token)])
                    appliedrule = 'brasan'

        #print('new ', new)
        dfp = self.tcP.copy()
        #print('dfp ', dfp)
        #print('self.newP ', self.tcP)
        cols_with_1 = dfp.columns[dfp.loc[token] > 0].tolist()
        cols_with_0 = dfp.columns[dfp.loc[token] == 0].tolist()
        #print('pcols_with_1 ', cols_with_1)
        #print('pcols_with_0 ', cols_with_0)
        for c in cols_with_1:
            #print('c ', token, c)
            if c in pre.columns:
                f = pre.loc[token,c]

                if f:
                    new.extend([NegBoxNeg(c),Box(c)])
                else:
                    new.extend([BoxNeg(c),Box(c)])
                    appliedrule = 'penis'
        sv,urt = is_consistent(new)

        #print('new ', new)
        #print('sv ', sv,urt)
        return new,sv,urt,appliedrule

    def analyze_belief(self):
        prog = []
        present_prog = ''
        present_theory = []
        test_seq=''
        ptr = 0
        c=0
        while True:
            token = self.sequence_order[ptr]
            test_seq = f'{test_seq}{token}'
            th,sv,reason,ar = self.new_dong(test_seq,token)
            #print('sv,th ', sv,th)
            if sv: # th is consistent
                present_prog = f'{present_prog}{token}'
                #print('th ', th)
                #print('sv ', sv)
                print('status cons',present_prog,test_seq,reason,ar,token)
                ptr+=1
            else:
                prog.append(present_prog)
                print('status inco',present_prog,test_seq,reason,ar,token)
                present_prog=''
                test_seq=''
                # don't change ptr
                continue
            if ptr>=len(self.sequence_order):
                break
            if c>300:
                break
            c+=1
        print(present_prog)

    
    
    def new_belief(self,test_sequence,token,next,prev):
        new=[Box(token)]
        ff = fwd.loc[token,next]
        pp = pre.loc[token,next]
        FF = self.F.loc[token,next]
        PP = self.P.loc[token,next]

        extensions = self.FF[self.FF[token,:]==1]
        print('extensions ', extensions)

        # Prog 'tn' loc 'tn'
        if FF and ff:
            new.append(Box(token),Box(next))
        elif FF and not ff:
            new.append(NegBox(next))
        elif FF and not pp:
            new.append(BoxNeg(token))


        fwd,pre = self.make_FP(test_sequence)
        print('test_sequence ', test_sequence)
        #F,P = self.make_FP(self.sequence_order)
        print(fwd)
        aggregator = []
        belief_state = []
        if 0:
            for x in fwd.index:
                for y in fwd.columns:
                    f = fwd.loc[x,y]
                    F = self.F.loc[x,y]
                    p = pre.loc[x,y]
                    P = self.P.loc[x,y]
        for i,token in enumerate(test_sequence):
            next = test_sequence[i+1]
                
                

    def make_FP(self,sequence=None):
        return self.compute_order_relation_matrices(sequence or self.sequence_order)

    def compute_order_relation_matrices(self,trace, win_size=0):
        """
        Computes follows and precedes matrices over a symbolic trace.
        If win_size is set, performs rolling window aggregation.

        Parameters:
            trace: list of (c, m) tuples
            win_size: int or None — if set, compute rolling aggregation

        Returns:
            follows_df, precedes_df: DataFrames with values in [0.0, 1.0]
        """
        unique_states = sorted(set(trace))
        state_to_idx = {s: i for i, s in enumerate(unique_states)}
        n = len(unique_states)

        # Initialize count matrix
        follows_sum = np.zeros((n, n), dtype=int)
        num_windows = 0
        stride = 10

        if win_size:
            for i in range(0, len(trace) - win_size + 1, stride):
                window = trace[i:i+win_size]
                Gfollows = np.zeros((n, n), dtype=int)

                # Track positions where each symbol appears
                last_seen = {s: [] for s in unique_states}
                for pos, val in enumerate(window):
                    for earlier in unique_states:
                        if earlier != val and any(p < pos for p in last_seen[earlier]):
                            i_idx = state_to_idx[earlier]
                            j_idx = state_to_idx[val]
                            follows[i_idx, j_idx] = 1
                    last_seen[val].append(pos)

                follows_sum += follows
                num_windows += 1

            follows_avg = follows_sum / num_windows
        else:
            # No windowing — compute global binary matrix
            follows = np.zeros((n, n), dtype=int)
            last_seen = {s: [] for s in unique_states}
            for pos, val in enumerate(trace):
                for earlier in unique_states:
                    if earlier != val and any(p < pos for p in last_seen[earlier]):
                        i_idx = state_to_idx[earlier]
                        j_idx = state_to_idx[val]
                        follows[i_idx, j_idx] = 1
                last_seen[val].append(pos)

            follows_avg = follows

        # Transpose to get precedes
        precedes_avg = follows_avg.copy().T

        follows_df = pd.DataFrame(follows_avg, index=unique_states, columns=unique_states)
        precedes_df = pd.DataFrame(precedes_avg, index=unique_states, columns=unique_states)

        return follows_df, precedes_df


    def build_adj_from_seq(self,seq):
        """Adjacency counts from a raw sequence."""
        def successors(df: pd.DataFrame, s):
            """Columns j where df[s, j] > 0."""
            row = df.loc[s]
            return row.index[row.gt(0)].tolist()

        def predecessors(df: pd.DataFrame, s):
            """Rows i where df[i, s] > 0."""
            col = df[s]
            return col.index[col.gt(0)].tolist()

        def seq_pairs(seq):
            """Adjacent pairs (a_i, a_{i+1}) from a sequence."""
            return list(zip(seq[:-1], seq[1:]))

        pairs = seq_pairs(seq)
        if not pairs:
            return pd.DataFrame()
        a, b = zip(*pairs)
        return pd.crosstab(pd.Series(a, name='src'), pd.Series(b, name='dst'))

    def derive_fp(self):
        d = self.build_adj_from_seq(self.sequence_order)

    def transitive_closure(self, A: pd.DataFrame, include_reflexive: bool = False):
        """
        A: square DataFrame (0/1 or counts). Index and columns must match (same labels & order).
        Returns:
          F: reachability (after)  — j reachable from i via path length >= 1
          P: precedence  (before) — j occurs before i  (i.e., F.T)
        """
        # Align to common square index/cols just in case
        common = A.index.intersection(A.columns)
        A = A.loc[common, common].astype(bool)

        R = A.to_numpy(copy=True)  # boolean
        n = R.shape[0]

        # Floyd–Warshall for boolean reachability
        for k in range(n):
            R = R | (R[:, [k]] & R[[k], :])  # broadcasted boolean “via k”

        if include_reflexive:
            np.fill_diagonal(R, True)

        F = pd.DataFrame(R, index=common, columns=common)
        P = F.T
        return F.astype(int), P.astype(int)


    def transitive_closure_sparse(self, A: pd.DataFrame, include_reflexive: bool = False):
        common = A.index.intersection(A.columns)
        A = A.loc[common, common].astype(bool)
        X = csr_matrix(A.values.astype(np.int8))
        R = X.copy()

        # iteratively add paths until fixpoint: R := sign(R + R@R)
        # works well when graph diameter is small
        prev_nnz = -1
        while R.nnz != prev_nnz:
            prev_nnz = R.nnz
            R = (R + R @ R)
            R.data[:] = 1  # sign
            R.eliminate_zeros()

        if include_reflexive:
            R = R.tolil()
            R.setdiag(1)
            R = R.tocsr()

        F = pd.DataFrame(R.toarray(), index=common, columns=common)
        P = F.T
        return F.astype(int), P.astype(int)

    def reach_after_before(self, seq, allow_self=False, dtype=int):
        """
        seq: list/array/Series of symbols (strings/chars/etc.)
        F[i,j] = 1  iff some j occurs strictly after some i
        P = F.T
        allow_self=False -> F[i,i]=1 only if there's a later i (repeat)
        """
        # Alphabet in stable order of first appearance (any order is fine)
        syms = pd.Index(pd.unique(pd.Series(seq)))
        idx = {s: k for k, s in enumerate(syms)}
        n = len(syms)

        F = np.zeros((n, n), dtype=bool)      # after
        seen = np.zeros(n, dtype=bool)        # symbols seen in the suffix

        # scan right->left
        for x in reversed(seq):
            u = idx[x]
            # all symbols seen to the right occur after x
            if seen.any():
                F[u, :] |= seen
            # if you *don’t* want self edges unless there's a repeated x later:
            # seen[u] being False here ensures we won’t set F[u,u] yet.
            seen[u] = True

        if not allow_self:
            np.fill_diagonal(F, False)

        F_df = pd.DataFrame(F.astype(dtype), index=syms, columns=syms)
        P_df = F_df.T
        return F_df, P_df

    def reach_after_before_counts(self,seq):
        """
        seq: iterable of symbols (str/int/etc.)
        Returns:
          F_bool[i,j] = 1 iff ∃ positions t<u with seq[t]=i, seq[u]=j
          P_bool      = F_bool.T
          F_count[i,j]= # of ordered pairs (t,u), t<u, with seq[t]=i, seq[u]=j
        """
        syms = pd.Index(pd.unique(pd.Series(seq)))  # stable-order alphabet
        pos = {s:k for k,s in enumerate(syms)}
        n = len(syms)

        # frequency of symbols seen so far (left->right)
        seen = np.zeros(n, dtype=np.int64)
        C = np.zeros((n, n), dtype=np.int64)  # counts of pairs (i before j)

        for x in seq:
            j = pos[x]
            # all previously seen i contribute pairs (i -> current j)
            C[:, j] += seen
            # now include current j in the history
            seen[j] += 1

        F_bool = (C > 0).astype(int)
        P_bool = F_bool.T
        F_count = pd.DataFrame(C, index=syms, columns=syms)
        P_count = F_count.T
        return pd.DataFrame(F_bool, index=syms, columns=syms),\
                pd.DataFrame(P_bool, index=syms, columns=syms),\
                F_count,P_count


    def hop_stats(self,seq):
        syms = list(dict.fromkeys(seq))
        idx = {s: k for k, s in enumerate(syms)}
        n = len(syms)

        gaps = defaultdict(list)  # (i,j) -> list of hop lengths
        positions = defaultdict(list)

        # record positions of each symbol
        for pos, sym in enumerate(seq):
            positions[sym].append(pos)

        for i in syms:
            for j in syms:
                if i == j:
                    continue
                for p in positions[i]:
                    # all j after p
                    for q in positions[j]:
                        if q > p:
                            gaps[(i,j)].append(q - p)

        # aggregate
        stats = {}
        for pair, gs in gaps.items():
            stats[pair] = {
                "min": min(gs),
                "max": max(gs),
                "avg": sum(gs) / len(gs),
                "median": sorted(gs)[len(gs)//2],
                "count": len(gs)
            }
        return stats


data = list('abcdegefgabdte')


def canonical_program(sequence,objective=None):
    dic = {}
    execution_order = list(sequence)

    #print('rev ', execution_order)
    program_order = execution_order[::-1]
    y = Atom(objective) if objective else Atom('boll')
    for x in program_order:
        dic[x] = [y,Atom(x).precond()]
        y = Atom(x).precond()
    return Program(program_order,dic)

cp = canonical_program(data,'obes')

A = cp.build_adj_from_seq(data)
print(A)
f,p = cp.reach_after_before(data)
print('f\n', f)
print('p\n', p)
print('data ', data)
for i,t in enumerate(list(data)):
    try:
        n = list(data)[i+1]
    except IndexError:
        break
    ib = between_sets_last_a_before_b(data,t,n)
    print('t,n,ib ', t,n,ib)


def find_inconsistency(data,debug=0):
    data = list(data)
    cumul = []
    futur = []
    print('--')
    theory = []

    f,p = reach_after_before(data)

    for i, t in enumerate(tqdm(data, total=len(data))):
    #for i,t in enumerate(data):
        try:
            n = list(data)[i+1]
        except IndexError:
            break
        cumul.append(t)
        futur = list(data)[i+1:]
        ib = between_sets_last_a_before_b(data,t,n)
        #ic = between_sets_last_a_before_b(cumul,t,n)
        #fu = between_sets_last_a_before_b(futur,t,n)
        #print('t,n,ib ', t,n,ib,ic,fu)
        #print('ic ', ic)
        ic = [list(x) for x in ib if x]
        #print('ic ', ic)
        ff,pp = reach_after_before(data[:i])
        fff,ppp = reach_after_before(data[i:])
        if not ic:
            theory.append((t,[1,3]))
            theory.append((n,[1,2]))
        else:
            theory.append((t,[1,3]))
            for u in ic[0]:
                if u in pp.index and t in pp.columns and pp.loc[u,t]>0:
                    theory.append(('BRR', u,[3]))
                    theory.append((u,[2,4]))
                elif n in p.index and u in p.columns and p.loc[n,u]>0:
                    theory.append(('FIIIS',u,[3,4]))
                    theory.append((u,[2,4]))
                elif 0:
                    if 1:
                        pass
                    elif u in p.index and t in p.columns and p.loc[u,t]>0:
                        theory.append(('PAAA',u,[3,4]))
                        theory.append((u,[2,4]))
                    elif n in p.index and u in p.columns and p.loc[n,u]>0:
                        theory.append(('NXU',u,[3,4]))
                        theory.append((u,[2,4]))
                    elif t in fff.index and u in fff.columns and fff.loc[t,u]>0:
                        theory.append(('KUOO',u,[3,4]))
                        theory.append((u,[2,4]))
                    elif t in f.index and u in f.columns and f.loc[t,u]>0:
                        theory.append(('WWWww',u,[3,4]))
                        theory.append((u,[2,4]))
                else:
                    theory.append(('olmolomlo',u,[3,4]))
                    theory.append((u,[2,4]))
                    continue
                break
            
            theory.append((n,[1,2]))

    if debug:
        print('theory ', theory)
    return (len([x for x in theory if x[0]=='BRR']),len([x for x in theory if x[0]=='FIIIS']))

def get_f_p(data,*args,method=1):
    if method == 1:
        _, f = reach_bounded_hops_counts(data,1)
        p = f.T 
    elif method == 0:
        f,p = reach_after_before(data)
    return f,p




def find_inconsistency_alt(data,debug=0):
    data = list(data)
    cumul = []
    futur = []
    print('--')
    theory = []

    #_, f = reach_bounded_hops_counts(data,1)
    #p = f.T 

    #f,p = reach_after_before(data)
    f,p = get_f_p(data,1,method=0)

    onf_cnt=0
    for i, t in enumerate(tqdm(data, total=len(data))):
    #for i,t in enumerate(data):
        try:
            n = list(data)[i+1]
        except IndexError:
            break
        cumul.append(t)
        futur = list(data)[i+1:]
        ib = between_sets_last_a_before_b(data,t,n)
        #ic = between_sets_last_a_before_b(cumul,t,n)
        #fu = between_sets_last_a_before_b(futur,t,n)
        #print('t,n,ib ', t,n,ib,ic,fu)
        #print('ic ', ic)
        ic = [list(x) for x in ib if x]
        #print('ic ', ic)
        #ff,pp = reach_after_before(data[:i])
        ff,pp = get_f_p(data[:i],1,method=0)
        #fff,ppp = reach_after_before(data[i:],1)
        if not ic:
            onf_cnt+=1
            theory.append((t,[1,3]))
            theory.append((n,[1,2]))
        else:
            theory.append((t,[1,3]))
            isanbrr=0
            if 1:
                for ubb in ic:
                    for u in ubb:
                        if u in pp.index and t in pp.columns and pp.loc[u,t]>0:
                            theory.append(('BRR', u,[3]))
                            theory.append((u,[2,4]))
                            isanbrr=1
                            break
                    if isanbrr:
                        break


            isafiiis=0
            if 1:
                if not isanbrr:
                    for ubb in ic:
                        for u in ubb:
                            if n in p.index and u in p.columns and p.loc[n,u]>0:
                                theory.append(('FIIIS',u,[3,4]))
                                theory.append((u,[2,4]))
                                isafiiis=1
                                break
                        if isafiiis:
                            break
            if not isafiiis and not isanbrr:
                theory.append(('olmolomlo','u',[3,4]))
                theory.append(('u',[2,4]))
                    
            theory.append((n,[1,2]))

    if debug:
        print('theory ', theory)
    #return (len([x for x in theory if x[0]=='BRR']),len([x for x in theory if x[0]=='FIIIS']))
    asa = len([x for x in theory if x[0]=='BRR'])
    if asa == 0:
        asadiv = 1
    else:
        asadiv=asa
    cnx = len([x for x in theory if x[0]=='FIIIS'])
    olo = len([x for x in theory if x[0]=='olmolomlo'])
    return (asa/(asadiv+cnx),cnx/(asadiv+cnx),cnx/asadiv) # separation
    #return (asa/len(theory),cnx/len(theory),olo/len(theory)) # non-separation

if 0:
    print('hispl')
    find_inconsistency(data)
    1/0

    cp.printp()

'''
.99 .003
.98 .015 
.993 .007
.98 .015
.98 .016
.96 .033
.79 .20
.97 .032
.97 .027
.91 .08
.94 .06
'''




#3. extraction =================
def sketch_rule_f(a1,a2,fval,program=[],data=[]):
    ret = []
    if fval==1:
        print('a1,a2,data ', a1,a2,data)
        if a1 in data:
            ret.extend([Box(a1),Box(a2),NegBoxNeg(a1),NegBoxNeg(a2)])
        if a2 in data:
            ret.extend([Box(a2),NegBoxNeg(a1)])

    elif fval==0:
        if a1 in data:
            ret.extend([Box(a1),NegBox(a2),BoxNeg(a2)])
        if a2 in data:
            ret.extend([Box(a2),NegBox(a1)])
    else:
        print('hej')
    return ret

def rule_f2(a1,a2,fval,program=[],data=[]):
    ret = []
    if fval==1:
        print('a1,a2,data ', a1,a2,data)
        if a1 in data:
            ret.extend([Box(a1),Box(a2),NegBoxNeg(a1),NegBoxNeg(a2)])
        if a2 in data:
            ret.extend([Box(a2),NegBoxNeg(a1)])

    elif fval==0:
        if a1 in data:
            ret.extend([Box(a1),NegBox(a2),BoxNeg(a2)])
        if a2 in data:
            ret.extend([Box(a2),NegBox(a1)])
    else:
        print('hej')
    return ret

def extract_modal_theory(sequence, rules, programme, fwd, prec=None):
    theory = []
    prog_execution_order = programme.program_order[::-1]
    for i,token in enumerate(sequence):
        if i>=len(sequence)-1:
            break
        next = sequence[i+1]
        print('token,next ', token,next)
        forw = fwd if isinstance(fwd,int) else \
            fwd[prog_execution_order.index(token),prog_execution_order.index(next)]
        for rule in rules:
            formulas = rule(token,next,forw,programme, [token,next])
            theory.extend(formulas)
    return theory

def extract_belief(program_sequence,test_sequence=None,F=None,P=None):
    if not test_sequence:
        F,P = make_FP(program_sequence)




    






#4. consistency =================

def is_consistent(formulas):
    box_atoms = {}
    negbox_atoms = {}
    boxneg_atoms = {}
    negboxneg_atoms = {}

    for f in formulas:
        atoms = f.atoms

        if isinstance(f, Box):
            if atoms in negbox_atoms:
                return False, (f, negbox_atoms[atoms])
            box_atoms[atoms] = f

        elif isinstance(f, NegBox):
            if atoms in box_atoms:
                return False, (f, box_atoms[atoms])
            negbox_atoms[atoms] = f

        elif isinstance(f, BoxNeg):
            if atoms in negboxneg_atoms:
                return False, (f, negboxneg_atoms[atoms])
            boxneg_atoms[atoms] = f

        elif isinstance(f, NegBoxNeg):
            if atoms in boxneg_atoms:
                return False, (f, boxneg_atoms[atoms])
            negboxneg_atoms[atoms] = f

    return True, None


def all_maximally_consistent_subsets(S):
    max_consistent_sets = []
    n = len(S)

    # Try all non-empty subsets
    for i in range(1, n+1):
        for subset in itertools.combinations(S, i):
            if is_consistent(subset):
                # Check if it's maximal: no existing superset in result
                is_maximal = True
                for other in max_consistent_sets:
                    if set(subset) < set(other):  # proper subset
                        is_maximal = False
                        break
                if is_maximal:
                    # Remove all sets that are subsets of this one
                    max_consistent_sets = [
                        s for s in max_consistent_sets if not set(s) < set(subset)
                    ]
                    max_consistent_sets.append(subset)

    return max_consistent_sets

#print('all_maximally_consistent_subsets(y) ', all_maximally_consistent_subsets(y))

def one_maximally_consistent_subset(S):
    result = []
    for f in S:
        sv,_ = is_consistent(result + [f])
        if sv:
            result.append(f)
    return result
#print('one_maximally_consistent_subset(y) ', one_maximally_consistent_subset(y))

def are_consistent(f1, f2):
    lists = 0
    if isinstance(f1,list) and isinstance(f2,list):
        lists = 1
        for g1 in f1:
            for g2 in g2:
                svar = are_consistent(g1,g2)
                if not svar:
                    return False
    elif isinstance(f1,list):
        lists = 1
        for g1 in f1:
            svar = are_consistent(g1,f2)
            if not svar:
                return False
    elif isinstance(f2,list):
        lists = 1
        for g2 in f2:
            svar = are_consistent(f1,g2)
            if not svar:
                return False
    if lists:
        return True # reaching here with lists means that all checks have been made.

    # Get their modal types and atoms
    if f1.op != f2.op:
        return True
    atoms1 = f1.atoms
    atoms2 = f2.atoms

    # Modal conflict: □φ vs ¬□φ
    if (
        isinstance(f1, Box) and isinstance(f2, NegBox) or
        isinstance(f1, NegBox) and isinstance(f2, Box)
    ):
        return atoms1 != atoms2

    # Modal conflict: □¬φ vs ¬□¬φ
    if (
        isinstance(f1, BoxNeg) and isinstance(f2, NegBoxNeg) or
        isinstance(f1, NegBoxNeg) and isinstance(f2, BoxNeg)
    ):
        return atoms1 != atoms2

    # Otherwise consistent
    return True

def matches_schema(formulas, schema):
    for f in formulas:
        for g in schema.formulas:
            subst = match_modal_formulas(f, g)
            print('subst,f,g ', subst,f,g)
            if subst:
                # Apply subst to rest of schema
                if all(
                    any(match_modal_formulas(f2, apply_substitution(sch_f, subst))
                        for f2 in formulas)
                    for sch_f in schema.formulas
                ):
                    return True
    return False
def match_modal_formulas(f, g):
    # Return substitution {Variable: Atom} if match succeeds
    if type(f) != type(g):
        return None
    if len(f.atoms) != len(g.atoms):
        return None

    subst = {}
    for a1, a2 in zip(sorted(f.atoms, key=str), sorted(g.atoms, key=str)):
        if isinstance(a2, Variable):
            if a2 in subst:
                if subst[a2] != a1:
                    return None
            else:
                subst[a2] = a1
        elif a1 != a2:
            return None
    return subst

if 0:
    print('--------------------')
    a = Atom("a")
    b = Atom("b")
    c = Atom("c")

    formulas = [
        Box(a),
        Box(b),
        BoxNeg(c),
        Box(c)
    ]

    x = Variable("x")
    schema = Conjunction(Box(x), BoxNeg(x))

    print(matches_schema(formulas, schema))  # ✅ True — c matches x
    print('--------------------')

def matches_schema_consistently(formula, schema):
    vars_in_schema = set()
    for f in schema.formulas:
        vars_in_schema.update(a for a in f.atoms if isinstance(a, Variable))
    
    #print('\nvars_in_schema ', vars_in_schema)

    formula_atoms = list(formula.atoms)
    #print('formula_atoms ', formula_atoms)
    if not vars_in_schema:
        return is_consistent([formula] + schema.formulas)

    # Try all possible substitutions (cartesian product)
    for atoms_choice in itertools.product(formula_atoms, repeat=len(vars_in_schema)):
        subst = dict(zip(vars_in_schema, atoms_choice))
        #print('subst ', subst)
        grounded_schema = [apply_substitution(f, subst) for f in schema.formulas]
        #print('grounded_schema ', grounded_schema)
        #print('formula ', formula)
        #print('are_consistent(formula,grounded_schema) ', are_consistent(formula,grounded_schema))
        svar,reason = is_consistent([formula] + grounded_schema)
        if svar:
            return True

    return False

if 1:
    print('********')
    x = Variable("x")

    schema = Conjunction(Box(x), BoxNeg(x))

    print(matches_schema_consistently(Box(Atom("a")), schema))       # ✅ True
    print(matches_schema_consistently(NegBox(Atom("a")), schema))    # ❌ False
    print(matches_schema_consistently(BoxNeg(Atom("a")), schema))    # ✅ True
    print(matches_schema_consistently(NegBoxNeg(Atom("a")), schema)) # ❌ False
    print('--------------------')

class Filter:
    def __init__(self,*fmls):
        self.filter = {}
        for i,f in enumerate(fmls):
            self.filter[i] = f
    def classify_atom_position(self,mcs):
        positions = defaultdict(list)
        for f in mcs:
            for i,g in self.filter.items():
                if are_consistent(f,g):
                    posistions[f].append(i)
    def pr(self):
        for i,f in self.filter.items():
            print('i,f ', i,f)
    def match(self,mcs):
        if isinstance(mcs,ModalFormula):
            mcs = [mcs]
        matches = {}
        for m in mcs:
            matchdic={}
            for i,f in self.filter.items():
                sv = matches_schema_consistently(m,Conjunction(*f))
                if sv:
                    if m in matchdic:
                        matchdic[m].append(i)
                    else:
                        matchdic[m]=[i]
            for i,p in matchdic.items():
                print(f'{i}\t{p}\t{[self.filter[x] for x in p]}')
            matches[m] = p
        common = set(matches[mcs[0]]).intersection(*[matches[x] for x in mcs[1:]])
        print(Conjunction(*mcs),common)


class Position(Filter):
    def __init__(self,*patterns,op=None):
        self.dic = {}
        x = Variable("x")
        fmls = []
        for i,p in enumerate(patterns):
            #self.dic[i] = [b(x,op=op) for b in p]
            fmls.append([b(x,op=op) for b in p])

        Filter.__init__(self,*fmls)
        print('fmls ', fmls)

'''
N𝑝 ∧ ¬N¬𝑝
N¬𝑝 ∧ ¬N𝑝
¬N𝑝 ∧ ¬N¬𝑝
N𝑝 ∧ N¬𝑝
'''
kpos = Position([Box,NegBoxNeg],[BoxNeg,NegBox],[NegBox,NegBoxNeg],[Box,BoxNeg])
kpos.pr()
print('type(kpos) ', type(kpos))
kpos.match(Box('p'))
kpos.match([Box('q'),BoxNeg('q')])


def classify_atom_position(mcs,filter):
    status = defaultdict(set)
    for f in mcs:
        if isinstance(f, Box):
            for a in f.atoms:
                status[a].add("N(p)")
        elif isinstance(f, BoxNeg):
            for a in f.atoms:
                status[a].add("N(~p)")
        elif isinstance(f, NegBox):
            for a in f.atoms:
                status[a].add("~N(p)")
        elif isinstance(f, NegBoxNeg):
            for a in f.atoms:
                status[a].add("~N(~p)")

    classifications = {}

    for atom, flags in status.items():
        # Normalize to easier pattern matching
        flags = frozenset(flags)

        if flags == {"N(p)", "~N(~p)"}:
            classifications[atom] = "obligatory"
        elif flags == {"N(~p)", "~N(p)"}:
            classifications[atom] = "forbidden"
        elif flags == {"~N(p)", "~N(~p)"}:
            classifications[atom] = "permitted"
        elif flags == {"N(p)", "N(~p)"}:
            classifications[atom] = "inconsistent"
        else:
            classifications[atom] = f"unclassified: {flags}"

    return classifications


if 1:
    print('test 1 ---------------')
    data = 'abcdefg'
    F = np.zeros((len(data),len(data)))
    for j in range(len(data)):
        if j>0:
            F[j-1,j]=1.
    print(F)

    cp = canonical_program(data,'obes')

    cp.printp()
    seq = 'acbee'
    u = extract_modal_theory(seq,[rule_f2],cp,F)
    print('u ', u)
    y = list(set(u))
    print('y ', y)
    print('one_maximally_consistent_subset(y) ', one_maximally_consistent_subset(y))


if 2:
    print('test 2 ---------------')

    print('seq to program')
    data = 'rdpclmegthksuaxoqyvrdlmethsaxoqyvrpclmgthkuaxoyvzw'
    #data = 'oqyvrdlmethsaxoqyvrpc'
    #data = 'rdpclmegthksuaxoqyvrdlmethsa'
    data = 'egthksuaxoqyvrdlmethsa'
    data = 'abcdabdurt'
    cp = canonical_program(data,'getmilk')
    cp.printp()
    #f,p = make_FP(data)
    #cp.extract_belief('clme')
    test = 'clme'
    #cp.new_dong(test,'m')
    #cp.analyze_belief()
    cp.cumulative_belief()
    print('data ', data)

if 1:
    data = 'abcdefbg'
    f,p = reach_after_before(list(data))
    print('f \n', f)
    print('p \n', p)


def encode(listofpairs):
    first = 'abcdefghijkmnopqrstuvwxyz'
    code = ''
    for p in listofpairs:
        code = f'{code}{first[p[0]]}{p[1]}'
        #code = f'{code}{first[p[0]]}'
    return code
def reconstruct_eam(data):
    data = encode(data)
    m = re.findall('([a-z][0-9]*)',data)
    if not  m:
        1/0
    enc = []
    last = 0
    for g in m:
        if g == last:
            continue
        enc.append(g)
        last = g
        
    cp = canonical_program(enc,'getmilk')
    
    cb = find_inconsistency_alt(data)
    #cb = find_inconsistency(data)
    print('cb ', cb)
    return cb
    


