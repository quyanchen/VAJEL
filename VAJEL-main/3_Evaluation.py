from pathlib import Path
import subprocess
import numpy as np

ALPHA = 0.1
BETA = 0.9
EXPEND = 0.1
dataset = 'k14'
# =========================
node_embedding_file = Path(f"/VAJEL-master/result/{dataset}_stke_n.emb.npy")
attr_embedding_file = Path(f"/VAJEL-master/result/{dataset}_stke_a.emb.npy")
node_attr_file      = Path(f"/VAJEL-master/data/{dataset}_stke.node")

edge_list_file      = Path(f"/VAJEL-master/STKE_PPI/{dataset}_cc_tt.txt")
node_name_file      = Path(f"/VAJEL-master/STKE_PPI/{dataset}_STKE_node.txt") 

golden_file         = Path("golden_standard.txt")

result_dir = Path("result")
result_dir.mkdir(parents=True, exist_ok=True)

out_results_file = result_dir / f"results{dataset}.txt"
out_emb_txt      = result_dir / f"{dataset}_embeddings.txt"
out_named_emb    = result_dir / f"combined_{dataset}_n_emb.txt"
out_edge_sim     = result_dir / f"{dataset}_attr_sim.txt"
out_complex_file = result_dir / f"final_{dataset}_attr_output"

CONVERT_EXE = Path("ConvertPPI.exe")
MINING_EXE  = Path("Mining_Cliques.exe")

protein_out_file = Path("protein.temp")
cliques_file     = Path("cliques")
ppi_pair_file    = Path("ppi.pair")
ppi_matrix_file  = Path("ppi.matrix")


def load_array_auto(p: Path) -> np.ndarray:

    try:
        arr = np.load(p, allow_pickle=False)
        if hasattr(arr, "files"):
            raise ValueError(f"{p} looks like .npz; please specify which array key to load.")
        return arr
    except Exception:
        return np.loadtxt(p)


def read_edges(edge_path: Path):
    edges = []
    with edge_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split()
            edges.append((a, b))
    return edges


def read_node_names(node_path: Path):
    with node_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def combine_embeddings(node_emb: np.ndarray,
                       attr_emb: np.ndarray,
                       node_attr_pairs: np.ndarray,
                       alpha: float,
                       beta: float) -> np.ndarray:

    n_nodes, dim = node_emb.shape

    if node_attr_pairs.ndim != 2 or node_attr_pairs.shape[1] != 2:
        raise ValueError("node_attr_pairs must be of shape [num_pairs, 2]")

    node_idx = node_attr_pairs[:, 0].astype(int)
    attr_idx = node_attr_pairs[:, 1].astype(int)

    valid = (
        (node_idx >= 0) & (node_idx < n_nodes) &
        (attr_idx >= 0) & (attr_idx < attr_emb.shape[0])
    )
    if not np.all(valid):
        bad = np.count_nonzero(~valid)
        print(f"[Warn] Found {bad} out-of-bound (node, attr) pairs; they will be skipped.")

    node_idx = node_idx[valid]
    attr_idx = attr_idx[valid]

    attr_sum = np.zeros((n_nodes, dim), dtype=float)
    np.add.at(attr_sum, node_idx, attr_emb[attr_idx])

    cnt = np.bincount(node_idx, minlength=n_nodes).astype(float)  
    cnt = np.maximum(cnt, 1.0)[:, None] 
    attr_avg = attr_sum / cnt

    combined = alpha * node_emb + beta * attr_avg
    return combined


def write_named_embeddings(node_names, emb: np.ndarray, out_path: Path):
    if len(node_names) != emb.shape[0]:
        raise ValueError(f"Node name lines ({len(node_names)}) != embedding rows ({emb.shape[0]})")

    with out_path.open("w", encoding="utf-8") as f:
        for name, vec in zip(node_names, emb):
            f.write(name + "\t" + "\t".join(f"{x:.8f}" for x in vec) + "\n")


def build_edge_cosine_sim(edge_path: Path, named_emb_path: Path, out_path: Path):
    vectors = {}
    with named_emb_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            name = parts[0]
            vec = np.asarray(list(map(float, parts[1:])), dtype=float)
            vectors[name] = vec

    edges = read_edges(edge_path)

    def cos_sim(v1: np.ndarray, v2: np.ndarray) -> float:
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    missing = 0
    with out_path.open("w", encoding="utf-8") as fo:
        for u, v in edges:
            v1 = vectors.get(u, None)
            v2 = vectors.get(v, None)
            if v1 is None or v2 is None:
                missing += 1
                continue
            fo.write(f"{u} {v} {cos_sim(v1, v2)}\n")

    if missing > 0:
        print(f"[Warn] {missing} edges skipped due to missing node vectors.")


def density_score(temp_set, matrix):
    if len(temp_set) < 2:
        return 0.0
    temp_density_score = 0.0
    for m in temp_set:
        for n in temp_set:
            if n != m and matrix[m, n] != 0:
                temp_density_score += matrix[m, n]
    temp_density_score = temp_density_score / (len(temp_set) * (len(temp_set) - 1))
    return temp_density_score


def merge_cliques(new_cliques_set, matrix):
    seed_clique = []
    while True:
        if len(new_cliques_set) >= 2:
            seed_clique.append(new_cliques_set[0])

            temp_cliques_set = []
            for i in range(1, len(new_cliques_set)):
                if len(new_cliques_set[i].intersection(new_cliques_set[0])) == 0:
                    temp_cliques_set.append(new_cliques_set[i])
                elif len(new_cliques_set[i].difference(new_cliques_set[0])) >= 3:
                    temp_cliques_set.append(new_cliques_set[i].difference(new_cliques_set[0]))

            cliques_set = []
            for s in temp_cliques_set:
                score = density_score(s, matrix)
                lst = list(s) + [score]
                cliques_set.append(lst)

            cliques_set.sort(key=lambda a: a[-1], reverse=True)

            new_cliques_set = []
            for item in cliques_set:
                new_cliques_set.append(set(item[:-1]))

        elif len(new_cliques_set) == 1:
            seed_clique.append(new_cliques_set[0])
            break
        else:
            break

    return seed_clique


def expand_cluster(seed_clique, all_protein_set, matrix, expand_thres):
    expand_set = []
    complex_set = []

    for instance in seed_clique:
        temp_set = set()
        for j in all_protein_set.difference(instance):
            temp_score = 0.0
            for n in instance:
                temp_score += matrix[n, j]
            temp_score /= len(instance)
            if temp_score >= expand_thres:
                temp_set.add(j)
        expand_set.append(temp_set)

    for i in range(len(seed_clique)):
        complex_set.append(seed_clique[i].union(expand_set[i]))

    return complex_set


def run_coan(edge_sim_file: Path, out_complex: Path):
    Dic_map = {}
    Node1, Node2, Weight = [], [], []
    all_node_index = set()

    with edge_sim_file.open("r", encoding="utf-8") as f, protein_out_file.open("w", encoding="utf-8") as f_protein_out:
        idx = 0
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            a, b, w = parts[0], parts[1], float(parts[2])
            Node1.append(a); Node2.append(b); Weight.append(w)

            if a not in Dic_map:
                Dic_map[a] = idx
                f_protein_out.write(a + "\n")
                all_node_index.add(idx)
                idx += 1
            if b not in Dic_map:
                Dic_map[b] = idx
                f_protein_out.write(b + "\n")
                all_node_index.add(idx)
                idx += 1

    Node_count = len(Dic_map)
    Map_dic = {v: k for k, v in Dic_map.items()}

    Adj = np.zeros((Node_count, Node_count), dtype=float)
    for a, b, w in zip(Node1, Node2, Weight):
        ia, ib = Dic_map[a], Dic_map[b]
        Adj[ia, ib] = w
        Adj[ib, ia] = w

    subprocess.run([str(CONVERT_EXE), str(edge_list_file), str(protein_out_file), str(ppi_pair_file), str(ppi_matrix_file)], check=True)
    mining_proc = subprocess.run([str(MINING_EXE), str(ppi_matrix_file), "1", "3", str(Node_count), str(cliques_file)], check=False)
    if (not cliques_file.exists()) or cliques_file.stat().st_size == 0:
        raise RuntimeError(f"Mining_Cliques failed (returncode={mining_proc.returncode}) and produced no cliques file.")

    cliques_set = []
    with cliques_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) <= 1:
                continue
            clique = [int(x) for x in parts[1:]]
            if len(clique) >= 2:
                cliques_set.append(clique)

    if len(cliques_set) == 0:
        raise RuntimeError("No cliques mined. Please check Mining_Cliques.exe output.")

    max_idx = max(max(c) for c in cliques_set)
    if max_idx >= Node_count:
        raise RuntimeError(f"Clique index out of range: max={max_idx}, Node_count={Node_count}. Check ConvertPPI input arguments and generated ppi.matrix.")

    scored = []
    for c in cliques_set:
        score = density_score(c, Adj)
        scored.append(c + [score])
    scored.sort(key=lambda a: a[-1], reverse=True)

    new_cliques_set = [set(item[:-1]) for item in scored]
    seed_clique = merge_cliques(new_cliques_set, Adj)

    expand_thres = EXPEND
    complex_set = expand_cluster(seed_clique, all_node_index, Adj, expand_thres)

    with out_complex.open("w", encoding="utf-8") as f:
        for comp in complex_set:
            f.write(" ".join(Map_dic[i] for i in comp) + "\n")


def evaluate(pred_file: Path, ref_file: Path):
    with pred_file.open("r", encoding="utf-8") as f:
        predicted_complex = [line.strip().split() for line in f if line.strip()]
    with ref_file.open("r", encoding="utf-8") as f:
        reference_complex = [line.strip().split() for line in f if line.strip()]

    predicted_num = len(predicted_complex)
    reference_num = len(reference_complex)

    matched_pred = 0
    for p in predicted_complex:
        best = 0.0
        sp = set(p)
        for r in reference_complex:
            sr = set(r)
            ov = sp & sr
            score = (len(ov) ** 2) / (len(sp) * len(sr)) if (len(sp) and len(sr)) else 0.0
            best = max(best, score)
        if best > 0.2:
            matched_pred += 1

    matched_ref = 0
    for r in reference_complex:
        best = 0.0
        sr = set(r)
        for p in predicted_complex:
            sp = set(p)
            ov = sp & sr
            score = (len(ov) ** 2) / (len(sp) * len(sr)) if (len(sp) and len(sr)) else 0.0
            best = max(best, score)
        if best > 0.2:
            matched_ref += 1

    precision = matched_pred / predicted_num if predicted_num else 0.0
    recall    = matched_ref / reference_num if reference_num else 0.0
    F1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    T_sum1, N_sum = 0.0, 0.0
    for r in reference_complex:
        sr = set(r)
        mx = 0
        for p in predicted_complex:
            sp = set(p)
            mx = max(mx, len(sr & sp))
        T_sum1 += mx
        N_sum += len(sr)
    Sn = (T_sum1 / N_sum) if N_sum else 0.0

    T_sum2, T_sum = 0.0, 0.0
    for p in predicted_complex:
        sp = set(p)
        mx = 0
        for r in reference_complex:
            sr = set(r)
            ov = sp & sr
            T_sum += len(ov)
            mx = max(mx, len(ov))
        T_sum2 += mx
    PPV = (T_sum2 / T_sum) if T_sum else 0.0
    Acc = float(np.sqrt(Sn * PPV)) if (Sn * PPV) >= 0 else 0.0

    return precision, recall, F1, Acc


def main():
    # 1) Load
    node_emb = load_array_auto(node_embedding_file)
    attr_emb = load_array_auto(attr_embedding_file)
    node_attr_pairs = np.loadtxt(node_attr_file, dtype=int)

    # 2) Combine embeddings (fixed alpha/beta)
    combined = combine_embeddings(node_emb, attr_emb, node_attr_pairs, ALPHA, BETA)
    np.savetxt(out_emb_txt, combined)
    print(f"[OK] Combined embeddings saved to: {out_emb_txt}")

    # 3) Write named embeddings
    node_names = read_node_names(node_name_file)
    write_named_embeddings(node_names, combined, out_named_emb)
    print(f"[OK] Named embeddings saved to: {out_named_emb}")

    # 4) Edge cosine sim
    build_edge_cosine_sim(edge_list_file, out_named_emb, out_edge_sim)
    print(f"[OK] Edge cosine similarities saved to: {out_edge_sim}")

    # 5) COAN mining
    run_coan(out_edge_sim, out_complex_file)
    print(f"[OK] Predicted complexes saved to: {out_complex_file}")

    # 6) Evaluate
    precision, recall, F1, Acc = evaluate(out_complex_file, golden_file)

    # 7) Write final result line
    with out_results_file.open("w", encoding="utf-8") as f:
        f.write("alpha\tbeta\tprecision\trecall\tF1\tAcc\n")
        f.write(f"{ALPHA:.2f}\t{BETA:.2f}\t{EXPEND:.2f}\t{precision:.4f}\t{recall:.4f}\t{F1:.4f}\t{Acc:.4f}\n")

    print(f"[OK] Metrics written to: {out_results_file}")
    print(f"alpha={ALPHA:.2f}, beta={BETA:.2f}, {EXPEND:.2f} | P={precision:.4f} R={recall:.4f} F1={F1:.4f} Acc={Acc:.4f}")


if __name__ == "__main__":
    main()
