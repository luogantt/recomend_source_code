#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 19:50:43 2020

@author: ledi
"""



import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from  deepfm import DeepFM
from feature_column import SparseFeat, DenseFeat, get_feature_names
from collections import OrderedDict
from tensorflow.python.keras.layers import Input
# if __name__ == "__main__":
data = pd.read_csv('./criteo_sample.txt')

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )
target = ['label']

# 1.Label Encoding for sparse features,and do simple Transformation for dense features
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

DEFAULT_GROUP_NAME = "default_group"
from collections import namedtuple
from tensorflow.python.keras.initializers import RandomNormal, Zeros





class Operate_Feat1():
    def __init__(self):
        self.sparse_dict={  'embedding_dim':4, 'use_hash':False,'dtype':"int32", 
            
            'feat_cat':'sparse',
            'embeddings_initializer':RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
           
             'embedding_name':None,'group_name':"default_group", 'trainable':True}
        self.dense_dict={'dimension':1, 'dtype':"float32", 'feat_cat':'dense',}
        
    def operate_sparse(self,some_data,name):
        sparse_dict1=self.sparse_dict
        sparse_dict1['vocabulary_size']=some_data.nunique()
        sparse_dict1['embedding_name'] =name
        return pd.Series(sparse_dict1)
    def operate_dense(self,dense_name):
        dense_dict1=self.dense_dict
        dense_dict1['name']=dense_name
        
        return pd.Series(dense_dict1)
        

d=Operate_Feat1()



sparse_list=[]
for p in sparse_features:
    d1=d.operate_sparse(data[p], p)
    sparse_list.append(d1.copy())

dense_list=[]
for q in dense_features:
    d2=d.operate_dense(q)
    print(d2)
    dense_list.append(d2.copy())


# fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4 )
#                         for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
#                       for feat in dense_features]

merge_list=sparse_list+dense_list
dnn_feature_columns = merge_list
linear_feature_columns = merge_list





# 构建输入层
def build_input_features(feature_columns, prefix=''):
    input_features = OrderedDict()
    for fc in feature_columns:
        if fc['feat_cat'] == 'sparse':
            input_features[fc['embedding_name']] = Input(
                shape=(1,), name=prefix + fc['embedding_name'], dtype=fc['dtype'])
        elif fc['feat_cat'] == 'dense':
            input_features[fc['name']] = Input(
                shape=(fc['dimension'],), name=prefix + fc['name'], dtype=fc['dtype'])
        # elif isinstance(fc, VarLenSparseFeat):
        #     input_features[fc.name] = Input(shape=(fc.maxlen,), name=prefix + fc.name,
        #                                     dtype=fc.dtype)
        #     if fc.weight_name is not None:
        #         input_features[fc.weight_name] = Input(shape=(fc.maxlen, 1), name=prefix + fc.weight_name,
        #                                                dtype="float32")
        #     if fc.length_name is not None:
        #         input_features[fc.length_name] = Input((1,), name=prefix + fc.length_name, dtype='int32')

        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return input_features


from itertools import chain

import tensorflow as tf

from feature_column import  get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from layers.core import PredictionLayer, DNN
from layers.interaction import FM
from layers.utils import concat_func, add_func, combined_dnn_input




fm_group=[DEFAULT_GROUP_NAME]
dnn_hidden_units=(128, 128)
l2_reg_linear=0.00001
l2_reg_embedding=0.00001
l2_reg_dnn=0
seed=1024
dnn_dropout=0
dnn_activation='relu'
dnn_use_bn=False
task='binary'

    

#构建模型的输入张量
features = build_input_features(
    merge_list)

print("#"*10)
print(features)
inputs_list = list(features.values())


# 构建线性张量
linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                l2_reg=l2_reg_linear)

group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                    seed, support_group=True)



#########################################################################################################

from feature_column import * 

feature_columns=dnn_feature_columns
l2_reg=1e-5
seed=1024
prefix=''
seq_mask_zero=True
support_dense=True
support_group=False
    

# 作者故意把简单的问题复杂化，让别人无法破译他的代码
# a=[1,2,3]
# a if 0 else [] ==>[]
# a if 1 else [] ==>[1,2,3]
# sparse_feature_columns = list(
#     filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []

sparse_feature_columns=[]
for fc in feature_columns:
    if  fc['feat_cat'] == 'sparse':
        sparse_feature_columns.append(fc)
        
    


# varlen_sparse_feature_columns = list(
#     filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []


'''
{'C1': <tensorflow.python.keras.layers.embeddings.Embedding at 0x7f5de6377910>,
 'C2': <tensorflow.python.keras.layers.embeddings.Embedding at 0x7f5de62dd1c0>}
'''



from keras.layers import Embedding
from tensorflow.python.keras.regularizers import l2
def create_embedding_dict(sparse_feature_columns ,seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    
    #将特征进行embedding ,输入维度是某个特征的种类数
    sparse_embedding = {}
    for feat in sparse_feature_columns:
        emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                        embeddings_initializer=feat.embeddings_initializer,
                        embeddings_regularizer=l2(l2_reg),
                        name=prefix + '_emb_' + feat.embedding_name)
        emb.trainable = feat.trainable
        sparse_embedding[feat.embedding_name] = emb

    return sparse_embedding

#embedding_matrix_dict是一个字典，key 是特征的名称，values 是某个特征的Embedding
embedding_matrix_dict = create_embedding_dict(sparse_feature_columns, l2_reg, seed, prefix=prefix,
                                                seq_mask_zero=seq_mask_zero)

def embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, return_feat_list=(),
                     mask_feat_list=(), to_list=False):
    group_embedding_dict = []
    for fc in sparse_feature_columns:
        feature_name = fc.embedding_name
        embedding_name = fc.embedding_name
        # if (len(return_feat_list) == 0 or feature_name in return_feat_list):
        if fc.use_hash:
            lookup_idx = Hash(fc.vocabulary_size, mask_zero=(feature_name in mask_feat_list))(
                sparse_input_dict[feature_name])
        else:
            
            # 模型输入层张量
            lookup_idx = sparse_input_dict[feature_name]
                                                       # 从输入层到embedding 层的映射
        group_embedding_dict.append(sparse_embedding_dict[embedding_name](lookup_idx))

    return group_embedding_dict
#这里面是从input 到embedding 层的映射 
group_sparse_embedding_dict = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns)

#获得dense的输入
dense_value_list = get_dense_input(features, feature_columns)
if not support_dense and len(dense_value_list) > 0:
    raise ValueError("DenseFeat is not supported in dnn_feature_columns")

sequence_embed_dict = varlen_embedding_lookup(embedding_matrix_dict, features, varlen_sparse_feature_columns)
group_varlen_sparse_embedding_dict = get_varlen_pooling_list(sequence_embed_dict, features,
                                                             varlen_sparse_feature_columns)
group_embedding_dict = mergeDict(group_sparse_embedding_dict, group_varlen_sparse_embedding_dict)
if not support_group:
    group_embedding_dict = list(chain.from_iterable(group_embedding_dict.values()))
# return group_embedding_dict, dense_value_list






#########################################################################################################

print('group_embedding_dict',group_embedding_dict)
print('dense_value_list',dense_value_list)

cc=[]
for k, v in group_embedding_dict.items():
    cc.append(v)
cc1=concat_func(cc[0], axis=1)

cc2=FM()(cc1)

# cc=[FM()(concat_func(v, axis=1))
#                       for k, v in group_embedding_dict.items() if k in fm_group]
fm_logit = add_func([cc2])

dnn_input = combined_dnn_input(list(chain.from_iterable(
    group_embedding_dict.values())), dense_value_list)
dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  dnn_use_bn, seed)(dnn_input)
dnn_logit = tf.keras.layers.Dense(
    1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_output)

final_logit = add_func([linear_logit, fm_logit, dnn_logit])

output = PredictionLayer(task)(final_logit)
model = tf.keras.models.Model(inputs=inputs_list, outputs=output)



feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 3.generate input data for model

train, test = train_test_split(data, test_size=0.2, random_state=2020)
train_model_input = {name:train[name] for name in feature_names}
test_model_input = {name:test[name] for name in feature_names}

# 4.Define Model,train,predict and evaluate
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
model.compile("adam", "binary_crossentropy",
              metrics=['binary_crossentropy'], )

history = model.fit(train_model_input, train[target].values,
                    batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
pred_ans = model.predict(test_model_input, batch_size=256)
print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
