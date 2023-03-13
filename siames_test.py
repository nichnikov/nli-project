import os
import keras
import fasttext
from utils import contrastive_loss
from src.texts_processing import TextsTokenizer

tokenizer = TextsTokenizer()

"""
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
"""

# keras.losses.custom_loss = contrastive_loss

ft_model = fasttext.load_model(os.path.join("models", "bss_cbow_lem.bin"))
print("ft_model:", ft_model)

nn_model = keras.models.load_model(os.path.join("models", "siamese_model_d2v_lstm2.h5"),
                                   custom_objects={'contrastive_loss': contrastive_loss})

"""
# keras.losses.custom_loss = contrastive_loss
# keras.losses.custom_objects = contrastive_loss
nn_model = keras.models.load_model(os.path.join("models", "siamese_model_d2v_lstm2.h5"))
"""
tx_pairs = [("какие есть штрафы за несвоевременную сдачу ефс?", "какой штраф будет за несвоевременную сдачу рвс?"),
            ("как правильно заполнить уведомление по транспортному налогу при оплате налога за 4 квартал 2022г.",
             "нужно ли подавать уведомление на оплату транспортного налога за 4 квартал 2022"),
            ("как правильно заполнить уведомление по транспортному налогу при оплате налога за 4 квартал 2022г.",
             "как заполнить уведомление по транспортному налогу"),
            ("Как подтвердить для налогового учета расходы по командировке", "налоговый учет командировочных расходов"),
            ("кбк для уведомления по страховым взносам НДФЛ, земельный налог",
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
             "отношении какого недвижимого имущества для расчета налога берется кадастровая стоимость? Организация ОСНО")
            ]

for tx1, tx2 in tx_pairs:
    lm_tx1 = tokenizer([tx1])[0]
    lm_tx2 = tokenizer([tx2])[0]

    v1_ = ft_model.get_sentence_vector(" ".join(lm_tx1))
    v2_ = ft_model.get_sentence_vector(" ".join(lm_tx2))

    v1 = v1_.reshape(1, 100, 1)
    v2 = v2_.reshape(1, 100, 1)

    score = nn_model.predict([v1, v2])
    print("\ntext1:", tx1, "\ntext2:", tx2, "\nscore: ", score[0][0])
