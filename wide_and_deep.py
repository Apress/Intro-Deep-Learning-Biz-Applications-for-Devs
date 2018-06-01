# to run : python wide_and_deep.py --method method
# example: python wide_and_deep.py --method deep
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler
from copy import copy

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Input, concatenate, Embedding, Reshape, Merge,
	Flatten, merge, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2, l1_l2

def cross_columns(x_cols):
"""simple helper to build the crossed columns in a pandas dataframe
"""
	crossed_columns = dict()
	colnames = ['_'.join(x_c) for x_c in x_cols]
	for cname,x_c in zip(colnames,x_cols):
		crossed_columns[cname] = x_c
	return crossed_columns

def val2idx(DF_deep,cols):
"""helper to index categorical columns before embeddings.
"""
	DF_deep = pd.concat([df_train, df_test])
	val_types = dict()
	for c in cols:
		val_types[c] = DF_deep[c].unique()
	val_to_idx = dict()
	for k, v in val_types.items():
		val_to_idx[k] = {o: i for i, o in enumerate(val_types[k])}
	for k, v in val_to_idx.items():
		DF_deep[k] = DF_deep[k].apply(lambda x: v[x])
	unique_vals = dict()
	for c in cols:
		unique_vals[c] = DF_deep[c].nunique()
	return DF_deep,unique_vals

def embedding_input(name, n_in, n_out, reg):
	inp = Input(shape=(1,), dtype='int64',name=name)
	return inp, Embedding(n_in, n_out, input_length=1,
		embeddings_regularizer=l2(reg))(inp)

def continous_input(name):
	inp = Input(shape=(1,), name=name)
	return inp, Reshape((1, 1))(inp)

def wide():
	target = 'cr'
	wide_cols = ["gender", "xyz_campaign_id", "fb_campaign_id", "age",
	"interest"]
	x_cols = (['gender', 'age'],['age', 'interest'])
	DF_wide = pd.concat([df_train,df_test])
	crossed_columns_d = cross_columns(x_cols)
	categorical_columns =list(DF_wide.select_dtypes(include=['object']).columns)
	wide_columns = wide_cols + crossed_columns_d.keys()

	for k, v in crossed_columns_d.iteritems():
		DF_wide[k] = DF_wide[v].apply(lambda x: '-'.join(x), axis=1)
	DF_wide = DF_wide[wide_columns + [target] + ['IS_TRAIN']]
	dummy_cols = [
		c for c in wide_columns if c in categorical_columns +
		crossed_columns_d.keys()]
	DF_wide = pd.get_dummies(DF_wide, columns=[x for x in dummy_cols])
	train = DF_wide[DF_wide.IS_TRAIN == 1].drop('IS_TRAIN', axis=1)
	test = DF_wide[DF_wide.IS_TRAIN == 0].drop('IS_TRAIN', axis=1)

	# sanity check: make sure all columns are in the same order
	cols = ['cr'] + [c for c in train.columns if c != 'cr']
	train = train[cols]
	test = test[cols]
	X_train = train.values[:, 1:]
	Y_train = train.values[:, 0]
	X_test = test.values[:, 1:]
	Y_test = test.values[:, 0]

	# WIDE MODEL
	wide_inp = Input(shape=(X_train.shape[1],), dtype='float32',
		name='wide_inp')
	w = Dense(1, activation="sigmoid", name = "wide_model")(wide_inp)
	wide = Model(wide_inp, w)
	wide.compile(Adam(0.01), loss='mse', metrics=['accuracy'])
	wide.fit(X_train,Y_train,nb_epoch=10,batch_size=64)
	results = wide.evaluate(X_test,Y_test)
	print "\n Results with wide model: %.3f" % results[1]


def deep():
	DF_deep = pd.concat([df_train,df_test])
	target = 'cr'
	embedding_cols = ["gender", "xyz_campaign_id", "fb_campaign_id",
		"age", "interest"]
	deep_cols = embedding_cols + ['cpc','cpco','cpcoa']
	DF_deep,unique_vals = val2idx(DF_deep, embedding_cols)
	train = DF_deep[DF_deep.IS_TRAIN == 1].drop('IS_TRAIN', axis=1)
	test = DF_deep[DF_deep.IS_TRAIN == 0].drop('IS_TRAIN', axis=1)
	n_factors = 5
	gender, gd = embedding_input('gender_in', unique_vals[
		'gender'], n_factors, 1e-3)
	xyz_campaign, xyz = embedding_input('xyz_campaign_id_in', unique_vals[
		'xyz_campaign_id'], n_factors, 1e-3)
	fb_campaign_id, fb = embedding_input('fb_campaign_id_in', unique_vals[
		'fb_campaign_id'], n_factors, 1e-3)
	age, ag = embedding_input('age_in', unique_vals[
		'age'], n_factors, 1e-3)
	interest, it = embedding_input('interest_in', unique_vals[
		'interest'], n_factors, 1e-3)
	# adding numerical columns to the deep model
	cpco, cp = continous_input('cpco_in')
	cpcoa, cpa = continous_input('cpcoa_in')
	X_train = [train[c] for c in deep_cols]
	Y_train = train[target]
	X_test = [test[c] for c in deep_cols]
	Y_test = test[target]
	# DEEP MODEL: input same order than in deep_cols:
	d = merge([gd, re, xyz, fb, ag, it], mode='concat')
	d = Flatten()(d)
	# layer to normalise continous columns with the embeddings
	d = BatchNormalization()(d)
	d = Dense(100, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(d)
	d = Dense(50, activation='relu',name='deep_inp')(d)
	d = Dense(1, activation="sigmoid")(d)
	deep = Model([gender, xyz_campaign, fb_campaign_id, age, interest,
		cpco, cpcoa], d)
	deep.compile(Adam(0.001), loss='mse', metrics=['accuracy'])
	deep.fit(X_train,Y_train, batch_size=64, nb_epoch=10)
	results = deep.evaluate(X_test,Y_test)
	print "\n Results with deep model: %.3f" % results[1]



def wide_deep():
	target = 'cr'
	wide_cols = ["gender", "xyz_campaign_id", "fb_campaign_id", "age",
		"interest"]
	x_cols = (['gender', 'xyz_campaign'],['age', 'interest'])
	DF_wide = pd.concat([df_train,df_test])
	crossed_columns_d = cross_columns(x_cols)
	categorical_columns =
		list(DF_wide.select_dtypes(include=['object']).columns)
	wide_columns = wide_cols + crossed_columns_d.keys()
	for k, v in crossed_columns_d.items():
		DF_wide[k] = DF_wide[v].apply(lambda x: '-'.join(x), axis=1)
		DF_wide = DF_wide[wide_columns + [target] + ['IS_TRAIN']]
	dummy_cols = [
		c for c in wide_columns if c in categorical_columns +
		crossed_columns_d.keys()]
	DF_wide = pd.get_dummies(DF_wide, columns=[x for x in dummy_cols])
	train = DF_wide[DF_wide.IS_TRAIN == 1].drop('IS_TRAIN', axis=1)
	test = DF_wide[DF_wide.IS_TRAIN == 0].drop('IS_TRAIN', axis=1)
	# sanity check: make sure all columns are in the same order
	cols = ['cr'] + [c for c in train.columns if c != 'cr']
	train = train[cols]
	test = test[cols]
	X_train_wide = train.values[:, 1:]
	Y_train_wide = train.values[:, 0]
	X_test_wide = test.values[:, 1:]
	DF_deep = pd.concat([df_train,df_test])
	embedding_cols = ['gender', 'xyz_campaign','fb_campaign_id', 'age',
	'interest']
	deep_cols = embedding_cols + ['cpco','cpcoa']
	DF_deep,unique_vals = val2idx(DF_deep,embedding_cols)
	train = DF_deep[DF_deep.IS_TRAIN == 1].drop('IS_TRAIN', axis=1)
	test = DF_deep[DF_deep.IS_TRAIN == 0].drop('IS_TRAIN', axis=1)
	n_factors = 5
	gender, gd = embedding_input('gender_in', unique_vals[
		'gender'], n_factors, 1e-3)
	xyz_campaign, xyz = embedding_input('xyz_campaign_id_in', unique_vals[
		'xyz_campaign_id'], n_factors, 1e-3)
	fb_campaign_id, fb = embedding_input('fb_campaign_id_in', unique_vals[
		'fb_campaign_id'], n_factors, 1e-3)
	age, ag = embedding_input('age_in', unique_vals[
		'age'], n_factors, 1e-3)
	interest, it = embedding_input('interest_in', unique_vals[
		'interest'], n_factors, 1e-3)
	# adding numerical columns to the deep model
	cpco, cp = continous_input('cpco_in')
	cpcoa, cpa = continous_input('cpcoa_in')
	X_train_deep = [train[c] for c in deep_cols]
	Y_train_deep = train[target]
	X_test_deep = [test[c] for c in deep_cols]
	Y_test_deep = test[target]
	X_tr_wd = [X_train_wide] + X_train_deep
	Y_tr_wd = Y_train_deep # wide or deep is the same here
	X_te_wd = [X_test_wide] + X_test_deep
	Y_te_wd = Y_test_deep # wide or deep is the same here
	#WIDE
	wide_inp = Input(shape=(X_train_wide.shape[1],), dtype='float32',
		name='wide_inp')
	#DEEP
	deep_inp = merge([ge, xyz, ag, fb, it, cp, cpa], mode='concat')
	deep_inp = Flatten()(deep_inp)
	# layer to normalise continous columns with the embeddings
	deep_inp = BatchNormalization()(deep_inp)
	deep_inp = Dense(100, activation='relu',
		kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(deep_inp)
	deep_inp = Dense(50, activation='relu',name='deep_inp')(deep_inp)
	#WIDE + DEEP
	wide_deep_inp = concatenate([wide_inp, deep_inp])
	wide_deep_out = Dense(1, activation='sigmoid',
		name='wide_deep_out')(wide_deep_inp)
	wide_deep = Model(inputs=[wide_inp, gender, age, xyz_campaign,
		fb_campaign_id,cpco, cpcoa],
		outputs=wide_deep_out)
	wide_deep.compile(optimizer=Adam(lr=0.001),loss='mse',
		metrics=['accuracy'])
	wide_deep.fit(X_tr_wd, Y_tr_wd, nb_epoch=50, batch_size=80)
	# wide_deep.optimizer.lr = 0.001
	# wide_deep.fit(X_tr_wd, Y_tr_wd, nb_epoch=5, batch_size=64)
	results = wide_deep.evaluate(X_te_wd, Y_te_wd)
	print "\n Results with wide and deep model: %.3f" % results[1]


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("--method", type=str, default="wide_deep",
	help="fitting method")
	args = vars(ap.parse_args())
	method = args["method"]
	df_train = pd.read_csv("train.csv")
	df_test = pd.read_csv("test.csv")
	df_train['IS_TRAIN'] = 1
	df_test['IS_TRAIN'] = 0
	if method == 'wide':
	wide()
	elif method == 'deep':
	deep()
	else:
	wide_deep()