#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 10:08:29 2023

@author: an
"""
import os
import keras
import fasttext
import time
import numpy as np
from keras.models import Model
from keras import backend as K
from keras.layers import Dense, Dropout, LSTM, Input
from src.texts_processing import TextsTokenizer

tokenizer = TextsTokenizer()



def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    # x = Flatten()(input)
    x = LSTM(250, input_shape=input_shape)(input)
    x = Dropout(0.1)(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


ft_model = fasttext.load_model(os.path.join("models", "bss_cbow_lem.bin"))
nn_model = keras.models.load_model(os.path.join("models", "siamese_model_ft_lstm_25e.h5"),
                                   custom_objects={'contrastive_loss': contrastive_loss})


tx_pairs = [("какие есть штрафы за несвоевременную сдачу ефс?", "какой штраф будет за несвоевременную сдачу рвс?"),
            ("как правильно заполнить уведомление по транспортному налогу при оплате налога за 4 квартал 2022г.",
             "нужно ли подавать уведомление на оплату транспортного налога за 4 квартал 2022"),
            ("как правильно заполнить уведомление по транспортному налогу при оплате налога за 4 квартал 2022г.",
             "как заполнить уведомление по транспортному налогу"),
            ("Как подтвердить для налогового учета расходы по командировке", "налоговый учет командировочных расходов"),
            ("кбк для уведомления по страховым взносам НДФЛ, земельный налог",
             "уведомление об исчисленных суммах налога кбк по страховым взносам"),
            ("кбк для уведомления земельный налог",
             "уведомление об исчисленных суммах налога кбк по страховым взносам"),
            ("как рассчитать транспортный налог на гибридный автомобиль",
             "как рассчитать транспортный налог по дорогим автомобилям"),
            ("Показывать ли мобилизованных сотрудников в отчете персонифицированные сведения?",
             "выплаты уволенного сотрудника включать в отчет персонифицированные сведения"),
            ("как заполнить новую форму 6 ндфл за 2022 год", "отпускные в новой форме 6-ндфл в 2021"),
            ("Нужно ли подавать пустой персонифицированный отчет если нет сотрудников",
             "если нет выплат, то надо ли сдавать персонифицированный отчет"),
            ("Здравствуйте, есть ли сроки подачи корректировочного уведомления по налогам?",
             "срок подачи уведомления по налогу на прибыль"),
            (
            "отношении какого недвижимого имущества для расчета налога берется кадастровая стоимость? Организация ОСНО",
            "расчет среднегодовой стоимости имущества имущество организации на усн"),
            ("расчет среднегодовой стоимости имущества имущество организации на усн",
             "отношении какого недвижимого имущества для расчета налога берется кадастровая стоимость? Организация ОСНО"),
            ("нужно ли сдавать декларацию по налогу на имущество за 2022 г., если налог рассчитывается с кадастровой стоимости?",
             "декларация по налогу на имущество с кадастровой стоимости за 2022"),
            ("нужно ли сдавать декларацию по налогу на имущество за 2022 г.",
             "декларация по налогу на имущество с кадастровой стоимости за 2022"),
            ("Подали уведомление в налоговую НДФЛ и страховые взносы. По страховым взносам ошибочно указали КБК ЕНП. При корректировке уведомления нужно также отражать НДФЛ и страховые взносы, или только страховые взносы с верными реквизитами?",
             "какие кбк указывать в уведомлении по страховым взносам и ндфл"),
            ("Подали уведомление в налоговую НДФЛ и страховые взносы.",
             "какие кбк указывать в уведомлении по страховым взносам и ндфл"),
            ("При корректировке уведомления нужно также отражать НДФЛ и страховые взносы, или только страховые взносы с верными реквизитами?",
             "какие кбк указывать в уведомлении по страховым взносам и ндфл"),
            (" По страховым взносам ошибочно указали КБК ЕНП",
             "какие кбк указывать в уведомлении по страховым взносам и ндфл"),
            ("ип оплачивает енп. статус платежа 01,верно?",
             "кбк фиксированных платежей ип в 2023")
            ]

k = 1 
for tx1, tx2 in tx_pairs:
    t = time.time()
    lm_tx1 = tokenizer([tx1])[0]
    lm_tx2 = tokenizer([tx2])[0]

    v1_ = ft_model.get_sentence_vector(" ".join(lm_tx1))
    v2_ = ft_model.get_sentence_vector(" ".join(lm_tx2))

    v1 = v1_.reshape(1, 100, 1)
    v2 = v2_.reshape(1, 100, 1)

    score = nn_model.predict([v1, v2])
    print("\n", k, "\ntext1:", tx1, "\ntext2:", tx2, "\nscore: ", score[0][0])
    print("working time:", time.time() - t)
    k += 1
