def autoencoder_egr(_x):

    #import numpy as np
    #import keras
    from keras.models import Sequential, Model
    from keras.layers import Dense, Input
    #from keras import optimizers
    #from keras.optimizers import Adam
    from sklearn.preprocessing import scale
    
    autoencoder = Sequential()
    autoencoder.add(Dense(16, activation='linear', input_shape=(51,)))
    autoencoder.add(Dense(8, activation='tanh', name="bottleneck"))
    autoencoder.add(Dense(16,  activation='relu'))
    autoencoder.add(Dense(51,  activation='sigmoid', name='reconstructedoutput')) 
    encoder = Model(autoencoder.input, autoencoder.get_layer('bottleneck').output)
    
    # set weights & bias
    
    
    encoded_data = encoder.predict(_x)
    decoded_output = autoencoder.predict(_x)
    return [encoded_data,decoded_output]
    
def predictor_egr(_x):

    #import numpy as np
    #import keras
    from keras.models import Sequential, Model
    from keras.layers import Dense, Input
    #from keras import optimizers
    #from keras.optimizers import Adam
    from sklearn.preprocessing import scale
    
    Clf1ab = Sequential()
    Clf1ab.add(Dense(16, activation='tanh', input_shape=(51,)))
    Clf1ab.add(Dense(8, activation='relu'))
    Clf1ab.add(Dense(16,  activation='sigmoid'))
    
    # set weights & bias
    
    
    y_pred_Clf1ab = Clf1ab.predict(_x)
    return [y_pred_Clf1ab]
