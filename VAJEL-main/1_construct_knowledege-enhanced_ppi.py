def load_protein_go(file_path):
    """
    读取蛋白质及其GO CC术语，返回一个字典：
    key: 蛋白质节点
    value: 对应的GO CC术语集合
    """
    protein_go = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            protein = parts[0]
            go_terms = set(parts[1:])
            protein_go[protein] = go_terms
    return protein_go

def filter_ppi(ppi_file, protein_go, output_file):
    """
    读取PPI网络，对每个蛋白质对：
      - 若两个蛋白质存在至少一个相同的GO CC术语，则保留该PPI边；
      - 若两个蛋白质均没有GO CC术语（包括在protein_go中不存在或者对应集合为空），也保留该边；
    并写入输出文件。
    """
    with open(ppi_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            # 去除末尾换行符，避免重复换行
            line = line.rstrip("\n")
            parts = line.split()
            if len(parts) < 2:
                continue
            protein1, protein2 = parts[0], parts[1]
            go1 = protein_go.get(protein1, set())
            go2 = protein_go.get(protein2, set())
            
            # 判断保留条件
            if (go1 & go2) or (not go1 and not go2):
                fout.write(line + "\n")

if __name__ == "__main__":
    # 文件路径（请根据实际情况修改）
    protein_go_file = r"dip_cc.txt"
    ppi_file = r"dip_tt.txt"
    output_file = r"dip_cc_tt.txt"
    
    protein_go = load_protein_go(protein_go_file)
    filter_ppi(ppi_file, protein_go, output_file)
    
    print("过滤后的PPI已保存到", output_file)
