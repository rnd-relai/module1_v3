def RTPv2model():
    # Need to import dependencies from Train Schema
    Tencoder = Dense(encoding_dim, activation="linear", input_shape=(input_dim,), use_bias = True, kernel_regularizer=WeightsOrthogonalityConstraint(encoding_dim, weightage=1., axis=0), kernel_constraint=UnitNorm(axis=0)) 
    Tdecoder = DenseTied(input_dim, activation="linear",  use_bias = False) #tied_to=encoder,

    Tautoencoder = Sequential()
    Tautoencoder.add(Tencoder)
    Tautoencoder.add(Tdecoder)
    
    return Tautoencoder
