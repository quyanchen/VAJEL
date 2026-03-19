import argparse
import csv
import os
import tempfile
from collections import Counter


def split_tokens(line):
    return [x for x in line.strip().split() if x]


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def parse_attidx_order(attidx_path):
    idx_to_go = {}
    with open(attidx_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = split_tokens(line)
            idx_to_go[int(parts[0])] = parts[1]
    order = [idx_to_go[i] for i in sorted(idx_to_go.keys())]
    return order, {go: i for i, go in enumerate(order)}


def normalize_go(go_id):
    s = str(go_id).strip()
    if s.startswith("GO:"):
        return s
    return "GO:" + s.zfill(7)


def parse_obo_namespace(obo_path):
    ns_map = {}
    alt_map = {}

    current = None

    def flush(term):
        if not term:
            return
        if term.get("is_obsolete", False):
            return
        tid = term.get("id")
        ns = term.get("namespace")
        if tid and ns:
            ns_map[tid] = ns
            for aid in term.get("alt_ids", []):
                alt_map[aid] = tid

    with open(obo_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line == "[Term]":
                flush(current)
                current = {"alt_ids": []}
                continue
            if not line:
                flush(current)
                current = None
                continue
            if current is None:
                continue
            if line.startswith("id:"):
                current["id"] = line.split("id:", 1)[1].strip()
            elif line.startswith("namespace:"):
                current["namespace"] = line.split("namespace:", 1)[1].strip()
            elif line.startswith("alt_id:"):
                current["alt_ids"].append(line.split("alt_id:", 1)[1].strip())
            elif line.startswith("is_obsolete:"):
                current["is_obsolete"] = line.split("is_obsolete:", 1)[1].strip().lower() == "true"

    flush(current)
    return ns_map, alt_map


def write_attidx(out_attidx, go_order, keep_unknown):
    with open(out_attidx, "w", encoding="utf-8") as f:
        if keep_unknown:
            f.write("#Attributes\t{}\n".format(len(go_order) + 1))
            f.write("0\tUNKNOWN_STKE\n")
            for i, go in enumerate(go_order, start=1):
                f.write("{}\t{}\n".format(i, go))
        else:
            f.write("#Attributes\t{}\n".format(len(go_order)))
            for i, go in enumerate(go_order):
                f.write("{}\t{}\n".format(i, go))


def main():
    parser = argparse.ArgumentParser(description="Prepare STKE_PPI data for VAJEL and reviewer experiments.")
    parser.add_argument("--stke-dir", default="STKE_PPI", help="Directory containing STKE files")
    parser.add_argument("--dataset", default="dip", help="Dataset prefix, e.g., bio/col/dip/k14")
    parser.add_argument("--out-prefix", default="data/dip_stke", help="Output prefix for .edge/.node/.attidx")
    parser.add_argument("--report-dir", default="result/stke_review", help="Report output directory")
    parser.add_argument("--obo", default="go-basic.obo", help="GO OBO file for namespace stats")
    parser.add_argument("--keep-unknown-col", action="store_true", help="Keep STKE column-0 unknown attribute")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    node_file = os.path.join(args.stke_dir, "{}_STKE_node.txt".format(args.dataset))
    net_file = os.path.join(args.stke_dir, "Network_{}_STKE.txt".format(args.dataset))
    attr_file = os.path.join(args.stke_dir, "Attribute_{}_STKE.txt".format(args.dataset))
    go_info_file = os.path.join(args.stke_dir, "{}_STKE_go_information_temp.txt".format(args.dataset))

    nodes = [x.strip() for x in read_lines(node_file) if x.strip()]
    go_order, go_to_idx = parse_attidx_order(args.dip_attidx)
    attr_lines = read_lines(attr_file)

    if len(attr_lines) != len(nodes):
        raise ValueError("Attribute lines ({}) != node count ({})".format(len(attr_lines), len(nodes)))

    first_attr = split_tokens(attr_lines[0])
    attr_cols = len(first_attr)
    expected_cols = len(go_order) + 1
    if attr_cols != expected_cols:
        raise ValueError(
            "Unexpected attribute columns: got {}, expected {} ({} GO + unknown col)".format(
                attr_cols, expected_cols, len(go_order)
            )
        )

    out_edge = args.out_prefix + ".edge"
    out_node = args.out_prefix + ".node"
    out_attidx = args.out_prefix + ".attidx"
    out_nodidx = args.out_prefix + ".nodidx"

    write_attidx(out_attidx, go_order, args.keep_unknown_col)

    # STKE mapping observed in this repo: col0 = unknown placeholder; col1..142 = dip.attidx 0..141
    col_to_attr = {}
    if args.keep_unknown_col:
        col_to_attr[0] = 0
        for c in range(1, len(go_order) + 1):
            col_to_attr[c] = c
        n_attrs = len(go_order) + 1
    else:
        for c in range(1, len(go_order) + 1):
            col_to_attr[c] = c - 1
        n_attrs = len(go_order)

    row_attrs = []
    annotated_nodes = 0
    total_pairs = 0
    attr_freq = Counter()

    with open(out_node, "w", encoding="utf-8") as fout:
        fout.write("#Nodes\t{}\n".format(len(nodes)))
        fout.write("#Attributes\t{}\n".format(n_attrs))
        for i, line in enumerate(attr_lines):
            vals = split_tokens(line)
            if len(vals) != attr_cols:
                raise ValueError("Attribute row {} has {} cols, expected {}".format(i, len(vals), attr_cols))
            present = []
            for c, v in enumerate(vals):
                if v == "1" and c in col_to_attr:
                    aidx = col_to_attr[c]
                    present.append(aidx)
                    fout.write("{}\t{}\n".format(i, aidx))
            row_attrs.append(set(present))
            if present:
                annotated_nodes += 1
            total_pairs += len(present)
            for aidx in present:
                attr_freq[aidx] += 1

    with open(out_nodidx, "w", encoding="utf-8") as f:
        for i, name in enumerate(nodes):
            f.write("{}\t{}\n".format(i, name))

    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tmp:
        tmp_path = tmp.name
        edge_count = 0
        with open(net_file, "r", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                vals = split_tokens(line)
                if len(vals) != len(nodes):
                    raise ValueError("Network row {} has {} cols, expected {}".format(i, len(vals), len(nodes)))
                for j in range(i + 1, len(vals)):
                    if vals[j] == "1":
                        tmp.write("{}\t{}\n".format(i, j))
                        edge_count += 1

    with open(out_edge, "w", encoding="utf-8") as fout, open(tmp_path, "r", encoding="utf-8") as fin:
        fout.write("#Nodes\t{}\n".format(len(nodes)))
        fout.write("#Edges\t{}\n".format(edge_count))
        for line in fin:
            fout.write(line)
    os.remove(tmp_path)

    mismatch_rows = []
    if os.path.exists(go_info_file):
        go_lines = read_lines(go_info_file)
        if len(go_lines) == len(nodes):
            for i, line in enumerate(go_lines):
                parts = split_tokens(line)
                go_terms = parts[1:] if len(parts) > 1 else []
                expected = {go_to_idx[t] for t in go_terms if t in go_to_idx}
                got = set(row_attrs[i])
                if args.keep_unknown_col:
                    got = {g - 1 for g in got if g > 0}
                else:
                    got = {g for g in got if g < len(go_order)}
                if got != expected:
                    mismatch_rows.append(i)

    ns_map = {}
    alt_map = {}
    if os.path.exists(args.obo):
        ns_map, alt_map = parse_obo_namespace(args.obo)

    term_ns_count = Counter()
    pair_ns_count = Counter()
    unknown_terms = 0
    for aidx in range(n_attrs):
        if args.keep_unknown_col and aidx == 0:
            term_ns_count["unknown_placeholder"] += 1
            pair_ns_count["unknown_placeholder"] += attr_freq.get(0, 0)
            continue

        go = go_order[aidx - 1] if args.keep_unknown_col else go_order[aidx]
        go_full = normalize_go(go)
        canonical = alt_map.get(go_full, go_full)
        ns = ns_map.get(canonical, "unknown")

        term_ns_count[ns] += 1
        pair_ns_count[ns] += attr_freq.get(aidx, 0)
        if ns == "unknown":
            unknown_terms += 1

    summary_csv = os.path.join(args.report_dir, "{}_stke_data_summary.csv".format(args.dataset))
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["dataset", args.dataset])
        w.writerow(["nodes", len(nodes)])
        w.writerow(["edges", edge_count])
        w.writerow(["attributes", n_attrs])
        w.writerow(["annotated_nodes", annotated_nodes])
        w.writerow(["annotated_node_ratio", annotated_nodes / float(len(nodes) if nodes else 1)])
        w.writerow(["avg_terms_per_node", total_pairs / float(len(nodes) if nodes else 1)])
        w.writerow(["total_node_attribute_pairs", total_pairs])
        w.writerow(["go_terms_unknown_namespace", unknown_terms])
        w.writerow(["mismatch_rows_vs_go_info", len(mismatch_rows)])
        for k, v in sorted(term_ns_count.items()):
            w.writerow(["terms_in_{}".format(k), v])
        for k, v in sorted(pair_ns_count.items()):
            w.writerow(["pairs_in_{}".format(k), v])

    freq_csv = os.path.join(args.report_dir, "{}_stke_term_frequency.csv".format(args.dataset))
    with open(freq_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["attr_index", "go_id", "frequency"])
        for aidx, freq in sorted(attr_freq.items(), key=lambda x: (-x[1], x[0])):
            if args.keep_unknown_col and aidx == 0:
                go = "UNKNOWN_STKE"
            else:
                go = go_order[aidx - 1] if args.keep_unknown_col else go_order[aidx]
            w.writerow([aidx, go, freq])

    mismatch_csv = os.path.join(args.report_dir, "{}_stke_mismatch_rows.csv".format(args.dataset))
    with open(mismatch_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "node_name"])
        for i in mismatch_rows:
            w.writerow([i, nodes[i]])

    cmd_file = os.path.join(args.report_dir, "{}_next_commands.txt".format(args.dataset))
    with open(cmd_file, "w", encoding="utf-8") as f:
        f.write("# 1) Train on converted STKE data\n")
        f.write(
            "python train_stke_single.py --dataset {} --output-prefix result/{}_stke --cpu-only\n".format(
                args.out_prefix, args.dataset
            )
        )
        f.write("\n# 2) Run hierarchy evaluation for reviewer response\n")
        f.write(
            "python go_hierarchy_eval.py --attidx {}.attidx --node {}.node "
            "--attr-emb result/{}_stke_a.emb.npy --attr-logstd result/{}_stke_a_sig.emb.npy "
            "--obo {} --namespaces bp,mf --out-dir result/go_eval_{}_stke\n".format(
                args.out_prefix,
                args.out_prefix,
                args.dataset,
                args.dataset,
                args.obo,
                args.dataset,
            )
        )

    print("Prepared VAJEL dataset:", out_edge, out_node, out_attidx)
    print("Node mapping file:", out_nodidx)
    print("Reports:", summary_csv, freq_csv, mismatch_csv)
    print("Suggested next commands:", cmd_file)


if __name__ == "__main__":
    main()
