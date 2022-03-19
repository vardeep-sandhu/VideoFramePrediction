# split = {'Train': ['11', '12', '13', '14', '15', '16', '17', '18','19', '20', '21', '23', '24', '25', '01', '04'],
#                              'Test' : ['22', '02', '03', '05', '06', '07', '08', '09', '10']}

# fulldataset = KTH_Dataset(directory='C://Users//aysha//Cuda Lab//KTH')
# subset_datasets = {}
# fulldataset.prepare_data()

# for split, ids in split.items():
#     idx = fulldataset.get_indices_for_persons(ids)
#     subset_datasets[split] = Subset(fulldataset, idx)
    
    
# train_dataloader=DataLoader(subset_datasets['Train'] , batch_size=16, shuffle=True)
    
# test_dataloader=DataLoader(subset_datasets['Test'] , batch_size=16, shuffle=True)


# # In[2]:


# print('==>>> total trainning batch number: {}'.format(len(train_dataloader)))
# print('==>>> total testing batch number: {}'.format(len(test_dataloader)))

# for seq, seq_target in train_dataloader:
#     print('--- Sample')
#     print('Input:  ', seq.shape)
#     print('Target: ', seq_target.shape)
#     break
