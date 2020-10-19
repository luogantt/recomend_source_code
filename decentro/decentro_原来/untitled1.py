

import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from  deepfm import DeepFM
from feature_column import SparseFeat, DenseFeat, get_feature_names

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

# 2.count #unique features for each sparse field,and record dense feature field name

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4 )
                       for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                      for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns




from itertools import chain

import tensorflow as tf

from feature_column import build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
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
    linear_feature_columns +dnn_feature_columns)

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
sparse_feature_columns = list(
    filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
varlen_sparse_feature_columns = list(
    filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []


'''
{'C1': <tensorflow.python.keras.layers.embeddings.Embedding at 0x7f5de6377910>,
 'C2': <tensorflow.python.keras.layers.embeddings.Embedding at 0x7f5de62dd1c0>}
'''
#embedding_matrix_dict是一个字典，key 是特征的名称，values 是某个特征的Embedding
embedding_matrix_dict = create_embedding_matrix(feature_columns, l2_reg, seed, prefix=prefix,
                                                seq_mask_zero=seq_mask_zero)


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

