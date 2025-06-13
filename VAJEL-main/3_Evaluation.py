import numpy as np
import os
node_embedding_file = f'bio_VAEJL_n.emb.npy' 
attr_embedding_file = f'bio_VAEJL_a.emb.npy'
node_attr_file = f'bio_VAEJL.node'
if not os.path.exists(node_attr_file):
	print(f"File not found: {node_attr_file}")
	continue
else:
	try:
		with open(node_attr_file, 'r') as file:
			lines = file.readlines()  
		with open(node_attr_file, 'w') as file:
			file.writelines(lines)

		print(f"Successfully removed the first two lines from {node_attr_file}.")
	except Exception as e:
		print(f"Error while processing file {node_attr_file}: {e}")
if not os.path.exists(node_embedding_file):
	print(f"Skipping {ii}: Missing file {node_embedding_file}.")
	continue
if not os.path.exists(attr_embedding_file):
	print(f"Skipping {ii}: Missing file {attr_embedding_file}.")
	continue
if not os.path.exists(node_attr_file):
	print(f"Skipping {ii}: Missing file {node_attr_file}.")
	continue
try:
	node_embeddings = np.load(node_embedding_file) 
	print(f"Loaded node embeddings from {node_embedding_file}")
except Exception as e:
	print(f"Error loading node embeddings: {e}")
	continue
try:
	attr_embeddings = np.load(attr_embedding_file)
	print(f"Loaded attribute embeddings from {attr_embedding_file}")
except Exception as e:
	print(f"Error loading attribute embeddings: {e}")
	continue
try:
	node_attr_pairs = np.loadtxt(node_attr_file, dtype=int) 
	print(f"Loaded node-attribute pairs from {node_attr_file}")
except Exception as e:
	print(f"Error loading node-attribute pairs: {e}")
	continue
with open('result/bio_result_VAEJL.txt', 'w') as result_file:
	result_file.write("alpha\tbeta\tprecision\trecall\tF1\tAcc\n")
	alpha = 0
	beta = 1- alpha
	combined_embeddings = np.zeros((node_embeddings.shape[0], node_embeddings.shape[1]))
	for node_idx, attr_idx in node_attr_pairs:
		if node_idx < 0 or node_idx >= node_embeddings.shape[0]:
			print(f"Warning: node index {node_idx} is out of bounds. Skipping.")
			continue
		if attr_idx < 0 or attr_idx >= attr_embeddings.shape[0]:
			print(f"Warning: attribute index {attr_idx} is out of bounds. Skipping.")
			continue

		# 加权平均
		combined_embeddings[node_idx] += (alpha * node_embeddings[node_idx] +
											beta * attr_embeddings[attr_idx])

	# 归一化结果
	attr_count = np.bincount(node_attr_pairs[:, 0], minlength=node_embeddings.shape[0])  # 这里不再减1
	combined_embeddings /= np.maximum(1, attr_count[:, np.newaxis])  # 保持维度一致
	np.savetxt('result/bio_VAEJL_embeddings.txt', combined_embeddings)
	print("Combined embeddings saved to 'bionew_embeddings.txt'")

	# 读取节点名
	with open(f'bio_VAEJL_node.txt', 'r') as f:
		bio_nodes = [line.strip() for line in f]

	with open('result/bio_VAEJL_embeddings.txt', 'r') as f:
		emb_vectors = [line.strip() for line in f]

	if len(bio_nodes) != len(emb_vectors):
		raise ValueError("两个文件的行数不匹配！")

	new_data = []

	for node, vector in zip(bio_nodes, emb_vectors):
		new_data.append([node] + vector.split())
	with open('result/combined_bio_VAEJL_n_emb.txt', 'w') as f:
		for row in new_data:
			f.write('\t'.join(row) + '\n')
	def cos_sim(vector1, vector2):
		dot_product = 0.0
		normA = 0.0
		normB = 0.0
		for a, b in zip(vector1, vector2):
			dot_product += a * b
			normA += a ** 2
			normB += b ** 2
		if normA == 0 or normB == 0:
			return 0.0 
		return dot_product / ((normA * normB) ** 0.5)

	for num in range(3, 4):
		str1 = f"bio_cc_tt.txt"  # Edge list file
		str2 = "result/combined_bio_VAEJL_n_emb.txt"  # Node embeddings file
		str3 = "result/bio_VAEJL_attr_sim.txt"  # Output file

		# Process subnet_graph file to get the nodes
		print(f"Processing: {str1}")
		with open(str1) as file1:
			node = set()  # Use a set for unique nodes
			edge_name_name = []

			for line in file1:
				temp1, temp2 = line.strip().split(' ')  # Split using tab
				node.add(temp1)
				node.add(temp2)
				edge_name_name.append((temp1, temp2))  # Store edges

		# Read vector file and map node names to their vectors
		vector = {}
		with open(str2) as file:
			for line in file:
				if not line.strip(): continue  # Skip empty lines
				parts = line.strip().split('\t')  # Split using tab
				node_name = parts[0]
				node_vector = list(map(float, parts[1:]))  # Convert remaining parts to floats
				vector[node_name] = np.array(node_vector)  # Store in dictionary

		# Process edges and calculate similarities
		with open(str3, 'w') as file2:
			for node_name1, node_name2 in edge_name_name:
				v1 = vector.get(node_name1)
				v2 = vector.get(node_name2)

				# Ensure vectors are valid and calculate similarity
				if v1 is not None and v2 is not None:
					result = cos_sim(v1, v2)
					file2.write(f"{node_name1} {node_name2} {result}\n")
				else:
					print(f"Warning: One of the nodes ({node_name1} or {node_name2}) does not have a vector.")

	def f_key(a):
		return (a[-1])


	def density_score(temp_set, matrix):
		temp_density_score = 0.
		for m in temp_set:
			for n in temp_set:
				if n != m and matrix[m, n] != 0:
					temp_density_score += matrix[m, n]

		temp_density_score = temp_density_score / (len(temp_set) * (len(temp_set) - 1))
		return temp_density_score


	def merge_cliques(new_cliques_set, matrix):
		seed_clique = []

		while (True):
			temp_cliques_set = []
			if len(new_cliques_set) >= 2:
				seed_clique.append(new_cliques_set[0])

				for i in range(1, len(new_cliques_set)):
					if len(new_cliques_set[i].intersection(new_cliques_set[0])) == 0:
						temp_cliques_set.append(new_cliques_set[i])
					elif len(new_cliques_set[i].difference(new_cliques_set[0])) >= 3:
						temp_cliques_set.append(new_cliques_set[i].difference(new_cliques_set[0]))

				cliques_set = []

				for i in temp_cliques_set:

					clique_score = density_score(i, matrix)
					temp_list = []
					for j in i:
						temp_list.append(j)

					temp_list.append(clique_score)
					cliques_set.append(temp_list)

				cliques_set.sort(key=f_key, reverse=True)

				new_cliques_set = []
				for i in range(len(cliques_set)):
					temp_set = set([])
					for j in range(len(cliques_set[i]) - 1):
						temp_set.add(cliques_set[i][j])
					new_cliques_set.append(temp_set)

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
			avg_node_score = density_score(instance, matrix)

			temp_set = set([])
			for j in all_protein_set.difference(instance):
				temp_score = 0.
				for n in instance:
					temp_score += matrix[n, j]
				temp_score /= len(instance)

				if (temp_score) >= expand_thres:
					temp_set.add(j)
			expand_set.append(temp_set)
		for i in range(len(seed_clique)):
			complex_set.append(seed_clique[i].union(expand_set[i]))

		return (complex_set)


	protein_out_file = "protein.temp"  # for each protein a index
	cliques_file = "cliques"
	ppi_pair_file = "ppi.pair"
	ppi_matrix_file = "ppi.matrix"

	if __name__ == "__main__":

		f = open("result/bio_VAEJL_attr_sim.txt", "r")
		f_protein_out = open(protein_out_file, "w")
		Dic_map = {}
		index = 0
		Node1 = []
		Node2 = []
		Weight = []
		All_node = set([])
		All_node_index = set([])

		for line in f:
			line = line.strip().split()
			if len(line) == 3:
				Node1.append(line[0])
				All_node.add(line[0])
				Node2.append(line[1])
				All_node.add(line[1])
				Weight.append(float(line[2]))
				if line[0] not in Dic_map:
					Dic_map[line[0]] = index
					f_protein_out.write(line[0] + "\n")
					All_node_index.add(index)
					index += 1
				if line[1] not in Dic_map:
					Dic_map[line[1]] = index
					f_protein_out.write(line[1] + "\n")
					All_node_index.add(index)
					index += 1
		Node_count = index
		f.close()
		f_protein_out.close()

		f.close()

		######dic_map to map_dic###########
		Map_dic = {}
		for key in Dic_map.keys():
			Map_dic[Dic_map[key]] = key

		# print Map_dic

		######bulid Adj_matrix###########

		Adj_Matrix = mat(zeros((Node_count, Node_count), dtype=float))

		if len(Node1) == len(Node2):

			for i in range(len(Node1)):
				if Node1[i] in Dic_map and Node2[i] in Dic_map:
					Adj_Matrix[Dic_map[Node1[i]], Dic_map[Node2[i]]] = Weight[i]
					Adj_Matrix[Dic_map[Node2[i]], Dic_map[Node1[i]]] = Weight[i]
		# print Adj_Matrix.shape[0]

		os.system(
			"ConvertPPI.exe " + "biogrid.txt" + " " + protein_out_file + " " + ppi_pair_file + " " + ppi_matrix_file)
		os.system(
			"Mining_Cliques.exe " + ppi_matrix_file + " " + "1" + " " + "3" + " " + str(Node_count) + " " + cliques_file)

		cliques_set = []
		f = open(cliques_file, "r")
		for line in f:
			temp_set = []
			line = line.strip().split()
			for i in range(1, len(line)):
				temp_set.append(int(line[i]))
			cliques_set.append(temp_set)

		f.close()
		avg_clique_score = 0.

		for instance in cliques_set:
			clique_score = density_score(instance, Adj_Matrix)
			avg_clique_score += clique_score
			instance.append(clique_score)
		# avg_clique_score /= len(cliques_set)
		cliques_set.sort(key=f_key, reverse=True)
		# print cliques_set

		new_cliques_set = []
		for i in range(len(cliques_set)):
			temp_set = set([])
			for j in range(len(cliques_set[i]) - 1):
				temp_set.add(cliques_set[i][j])
			new_cliques_set.append(temp_set)
		# print new_cliques_set
		# print len(new_cliques_set)

		seed_clique = merge_cliques(new_cliques_set, Adj_Matrix)
		# print seed_clique
		# print len(seed_clique)

		expand_thres = 0.3
		complex_set = expand_cluster(seed_clique, All_node_index, Adj_Matrix, expand_thres)
		print("##########output predicted complexes##########\n")
		final_file = open(f"result/final_bio_VAEJL_alpha{alpha}_attr_output", "w")

		for i in range(len(complex_set)):

			line = ""
			for m in complex_set[i]:
				line += Map_dic[m] + " "
			line += "\n"

			final_file.write(line)
		final_file.close()

		print("##########COAN completes############")
		
		import networkx as nx
		import numpy as np
		import matplotlib.pyplot as plt
		strr = f"result/final_bio_VAEJL_alpha{alpha}_attr_output"
		file = open(strr)
		file1 = open("golden_standard.txt")
		
		# g=nx.Graph()
		predicted_num = len(file.readlines())
		reference_num = len(file1.readlines())
		file.close()
		file1.close()
		
		file = open(strr)
		file1 = open("golden_standard.txt")
		reference_complex = []
		for j in file1:
			j = j.rstrip()
			j = j.rstrip('\n')
			complex_list = j.split(' ')
			reference_complex.append(complex_list)
		predicted_complex = []
		for i in file:
			i = i.rstrip()
			i = i.rstrip('\n')
			node_list = i.split(' ')
			predicted_complex.append(node_list)
		# precision
		number = 0
		c_number = 0
		row = 1
		for i in predicted_complex:
			overlapscore = 0.0
			for j in reference_complex:
				set1 = set(i)
				set2 = set(j)
				overlap = set1 & set2
				score = float((pow(len(overlap), 2))) / float((len(set1) * len(set2)))
				if (score > overlapscore):
					overlapscore = score
			if (overlapscore > 0.2):
				number = number + 1
			# print row,
			# print " ",
			row = row + 1
		# recall
		for i in reference_complex:
			overlapscore = 0.0
			for j in predicted_complex:
				set1 = set(i)
				set2 = set(j)
				overlap = set1 & set2
				score = float((pow(len(overlap), 2))) / float((len(set1) * len(set2)))
				if (score > overlapscore):
					overlapscore = score
			if (overlapscore > 0.25):
				c_number = c_number + 1
		# sn
		T_sum1 = 0.0
		N_sum = 0.0
		for i in reference_complex:
			max = 0.0
			for j in predicted_complex:
				set1 = set(i)
				set2 = set(j)
				overlap = set1 & set2
				if len(overlap) > max:
					max = len(overlap)
			T_sum1 = T_sum1 + max
			N_sum = N_sum + len(set1)
		# ppv
		T_sum2 = 0.0
		T_sum = 0.0
		for i in predicted_complex:
			max = 0.0
			for j in reference_complex:
				set1 = set(i)
				set2 = set(j)
				overlap = set1 & set2
				T_sum = T_sum + len(overlap)
				if len(overlap) > max:
					max = len(overlap)
			T_sum2 = T_sum2 + max
		
		# print "\n"
		print(number, predicted_num)  # matched predicted complex number
		# print c_number,reference_num# matched reference complex number
		precision = float(number / float(predicted_num))
		recall = float(c_number / float(reference_num))
		F1 = float((2 * precision * recall) / (precision + recall))
		Sn = float(T_sum1) / float(N_sum)
		PPV = float(T_sum2) / float(T_sum)
		Acc = pow(float(Sn * PPV), 0.5)
		print(strr)
		print(alpha)
		print(precision)
		print(recall)
		print(F1)
		print(Acc)
		# 将结果写入文件
		result_file.write(f"{alpha:.2f}\t{beta:.2f}\t{precision:.4f}\t{recall:.4f}\t{F1:.4f}\t{Acc:.4f}\n")
