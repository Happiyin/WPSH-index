# -*- coding: utf-8 -*-
# @Time    : 2020/6/5 17:16
# @Author  : Mr zhou
# @FileName: model_frame.py
# @Software: PyCharm
# @weixin    ：dayinpromise1314
#%%%%%%
def Network():
    from keras.models import Sequential, Model
    from keras.layers import Dense, Dropout, Activation, TimeDistributed, Flatten
    from keras.layers import LSTM, ConvLSTM2D
    from keras.models import Model
    from keras.layers import concatenate
    from keras.layers import Conv3D, MaxPooling2D,BatchNormalization,Conv2D
    from keras.utils import plot_model
    from keras import metrics, losses
    ########################1
    model1 = Sequential()
    model1.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         input_shape=(4, 301, 273, 1), padding='valid', return_sequences=True))
    model1.add(BatchNormalization())
    # model1.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='valid', return_sequences=True))
    # model1.add(BatchNormalization())
    # model1.add(Dropout(0.2))
    # model1.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='valid', return_sequences=True))
    # model1.add(BatchNormalization())
    model1.add(Dropout(0.2))
    model1.add(ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='valid', return_sequences=False))
    model1.add(BatchNormalization())
    #####################2
    model2 = Sequential()
    model2.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         input_shape=(4, 301, 273, 1), padding='valid', return_sequences=True))
    model2.add(BatchNormalization())
    # model2.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='valid', return_sequences=True))
    # model2.add(BatchNormalization())
    # model2.add(Dropout(0.2))
    # model2.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='valid', return_sequences=True))
    # model2.add(BatchNormalization())
    model2.add(Dropout(0.2))
    model2.add(ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='valid', return_sequences=False))
    model2.add(BatchNormalization())
    #####################3
    model3 = Sequential()
    model3.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         input_shape=(4, 301, 273, 1), padding='valid', return_sequences=True))
    model3.add(BatchNormalization())
    # model3.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='valid', return_sequences=True))
    # model3.add(BatchNormalization())
    # model3.add(Dropout(0.2))
    # model3.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='valid', return_sequences=True))
    # model3.add(BatchNormalization())
    model3.add(Dropout(0.2))
    model3.add(ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='valid', return_sequences=False))
    model3.add(BatchNormalization())
    #####################4
    model4 = Sequential()
    model4.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                          input_shape=(4, 301, 273, 1), padding='valid', return_sequences=True))
    model4.add(BatchNormalization())
    # model4.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='valid', return_sequences=True))
    # model4.add(BatchNormalization())
    # model4.add(Dropout(0.2))
    # model4.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='valid', return_sequences=True))
    # model4.add(BatchNormalization())
    model4.add(Dropout(0.2))
    model4.add(ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='valid', return_sequences=False))
    model4.add(BatchNormalization())
    #####################合并
    model_concat = concatenate([model1.output, model2.output, model3.output,model4.output], axis=-1)
    model_concat=Conv2D(filters=8, kernel_size=(2, 2), strides=(1, 1), activation='relu',
           kernel_initializer='glorot_uniform', padding='valid')(model_concat)
    model_concat=BatchNormalization()(model_concat)
    model_concat=Conv2D(filters=1, kernel_size=(2, 2), strides=(1, 1), activation='relu',
                kernel_initializer='glorot_uniform', padding='valid')(model_concat)
    model_concat=BatchNormalization()(model_concat)
    model_concat=MaxPooling2D()(model_concat)
    model_concat=Flatten()(model_concat)
    model_concat=Dropout(0.2)(model_concat)
    model_concat = Dense(32,activation='relu')(model_concat)
    model_concat = Dense(4)(model_concat)
    model = Model(inputs=[model1.input, model2.input, model3.input,model4.input], outputs=model_concat)
    model.compile(loss=losses.mean_squared_error, optimizer='adam',metrics=[metrics.mean_absolute_error])
    print(model.summary())
    plot_model(model, show_shapes=True, to_file='model.png')
    return model