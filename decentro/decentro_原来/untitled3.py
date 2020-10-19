#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 16:33:58 2020

@author: ledi

文章链接
https://www.kaggle.com/rajmehra03/a-detailed-explanation-of-keras-embedding-layer
"""

###A Detailed Explanation of Keras Embedding Layer
### ebedding 层的详细解释

在这篇文章中，我们将介绍keras的嵌入层。为此，我创建了一个仅包含3个文档的样本语料库，这足以解释keras嵌入层的工作。

词嵌入在各种机器学习应用程序中很有用
在开始之前，让我们浏览一下词嵌入的一些应用：

1）第一个吸引我的应用程序是在基于协同过滤的推荐系统中，我们必须通过分解包含用户项等级的效用矩阵来创建用户嵌入和电影嵌入。

要查看有关在Keras中使用词嵌入的基于CF推荐系统的完整教程，可以遵循我的这篇文章。

2）第二种用途是在自然语言处理及其相关应用程序中，我们必须为语料库文档中存在的所有单词创建单词嵌入。

这是我将在此内核中使用的术语。

因此，当我们想要创建将高维数据嵌入到低维向量空间中的嵌入时，可以使用Keras中的嵌入层。


import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
# %matplotlib inline  
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#nltk
import nltk

#stop-words
from nltk.corpus import stopwords
stop_words=set(nltk.corpus.stopwords.words('english'))

# tokenizing
from nltk import word_tokenize,sent_tokenize

#keras
import keras
from keras.preprocessing.text import one_hot,Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense , Flatten ,Embedding,Input
from keras.models import Model




# 这可以理解为三篇文章
sample_text_1="bitty bought a bit of butter"
sample_text_2="but the bit of butter was a bit bitter"
sample_text_3="so she bought some better butter to make the bitter butter better"

corp=[sample_text_1,sample_text_2,sample_text_3]
no_docs=len(corp)


#此后，所有唯一词都将由一个整数表示。 为此，我们使用Keras中的one_hot函数。 请注意，vocab_size被指定为足够大，以确保每个单词的唯一整数编码。
#注意一件重要的事情，即单词的整数编码在不同文档中保持不变。 例如，“黄油”在每个文档中都用31表示。

# 指定词向量的长度
vocab_size=50 
encod_corp=[]
for i,doc in enumerate(corp):
    encod_corp.append(one_hot(doc,50))
    
    # print(one_hot(doc,50))
    print("The encoding for document",i+1," is : ",one_hot(doc,50))
    
    

# length of maximum document. will be nedded whenever create embeddings for the words
maxlen=-1
for doc in corp:
    tokens=nltk.word_tokenize(doc)
    if(maxlen<len(tokens)):
        maxlen=len(tokens)
print("The maximum number of words in any document is : ",maxlen)

"""
Keras嵌入层要求所有单个文档的长度都相同。 因此，我们现在将较短的文档填充0。 因此，现在在Keras嵌入层中，“ input_length”将等于具有最大长度或最大单词数的文档的长度（即单词数）。

为了填充较短的文档，我使用Keras库中的pad_sequences函数。
"""
# now to create embeddings all of our docs need to be of same length. hence we can pad the docs with zeros.
pad_corp=pad_sequences(encod_corp,maxlen=maxlen,padding='post',value=0.0)
print("No of padded documents: ",len(pad_corp))

"""
现在所有文档的长度相同（填充后）。 因此，现在我们可以创建和使用嵌入了。
我将这些词嵌入8维向量中。
"""
# specifying the input shape
# input=Input(shape=(no_docs,maxlen),dtype='float64')


'''
shape of input. 
each document has 12 element or words which is the value of our maxlen variable.

'''
word_input=Input(shape=(maxlen,),dtype='float64')  

# creating the embedding
word_embedding=Embedding(input_dim=vocab_size,output_dim=8,input_length=maxlen)(word_input)

word_vec=Flatten()(word_embedding) # flatten
embed_model =Model([word_input],word_embedding) # combining all into a Keras model
embed_model.summary()

embed_model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),loss='binary_crossentropy',metrics=['acc']) 
# compiling the model. parameters can be tuned as always.

print(type(word_embedding))
print(word_embedding)

embeddings=embed_model.predict(pad_corp) # finally getting the embeddings.


print("Shape of embeddings : ",embeddings.shape)
print(embeddings)


embeddings=embeddings.reshape(-1,maxlen,8)
print("Shape of embeddings : ",embeddings.shape) 
print(embeddings)