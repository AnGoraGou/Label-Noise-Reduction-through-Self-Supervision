cluster_list = k_df.cl_label.unique()
cluster_list = sorted(cluster_list)

print(cluster_list)
#df_k = k_df.loc[k_df['cl_label'] == 4]
#ps = df_k['gt_label'].mode()

#print(ps.values)

for cluster_ in cluster_list:
    print(f'Working on cluster: {cluster_}')
    cl_df = k_df.loc[k_df['cl_label'] == cluster_]  # here SUBRAT change cl_label to gt_label
    ps = cl_df['gt_label'].mode().values
    #print(ps[0])
    k_df.loc[k_df['cl_label'] == cluster_, 'ps_label'] = ps[0]
    # print(f'pseudo label for cluster {cluster_} is {ps_label_c}')

print(k_df)

k_df['noise'] = k_df['gt_label'] ==k_df['ps_label']


print(f"Value counts of noise: {k_df['noise'].value_counts()}")
