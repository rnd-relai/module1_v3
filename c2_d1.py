def get_col_module(arg1='default'):
    if arg1=='relai_v2': 
        import RTP_v2_cols
        return RTP_v2_cols.col_list
     if arg1=='default': 
        print("nothing passed")        
        return 0
    return 0 # non-matching arg1 passed 
    

def input_data_module_s3(bucket ="relai.poc.data", folder_base = '/unified_Fx_D3_power_annotated/',file_type ='.csv' ):
    # if no arg passed, default data is for relai, fr testing phase .
    csv=file_type
     
    try:
        from d2 import get_vin_list #d2 is RelAI specific lib
        f_=get_vin_list(bucket=bucket, folder_base=folder_base, file_extn=csv)
        vin_pathlist = ['s3://{}/{}{}{}'.format(bucket, folder_base, vin, csv) for vin in f_]
    except: return 0 # error logging commands 
        
    return vin_pathlist


def RTPv2model(tech_params):
    # last tested on TF ver
    # tf.__version__  = '1.15.2'
    
    encoding_dim = tech_params['encoding_dim']
    input_dim = tech_params['input_dim']
    batch_size = tech_params['batch_size']
    shuffle = tech_params['shuffle']
    
    import tensorflow as tf
    from keras.models import Model, load_model
    from keras.layers import Input, Dense, Layer, InputSpec
    from keras.callbacks import ModelCheckpoint, TensorBoard
    from keras import regularizers, activations, initializers, constraints, Sequential
    from keras import backend as K
    from keras.constraints import UnitNorm, Constraint
    
    class DenseTied(Layer):
        def __init__(self, units,
                     activation=None,
                     use_bias=True,
                     kernel_initializer='glorot_uniform',
                     bias_initializer='zeros',
                     kernel_regularizer=None,
                     bias_regularizer=None,
                     activity_regularizer=None,
                     kernel_constraint=None,
                     bias_constraint=None,
                     tied_to=None,
                     **kwargs):
            self.tied_to = tied_to
            if 'input_shape' not in kwargs and 'input_dim' in kwargs:
                kwargs['input_shape'] = (kwargs.pop('input_dim'),)
            super().__init__(**kwargs)
            self.units = units
            self.activation = activations.get(activation)
            self.use_bias = use_bias
            self.kernel_initializer = initializers.get(kernel_initializer)
            self.bias_initializer = initializers.get(bias_initializer)
            self.kernel_regularizer = regularizers.get(kernel_regularizer)
            self.bias_regularizer = regularizers.get(bias_regularizer)
            self.activity_regularizer = regularizers.get(activity_regularizer)
            self.kernel_constraint = constraints.get(kernel_constraint)
            self.bias_constraint = constraints.get(bias_constraint)
            self.input_spec = InputSpec(min_ndim=2)
            self.supports_masking = True

        def build(self, input_shape):
            assert len(input_shape) >= 2
            input_dim = input_shape[-1]

            if self.tied_to is not None:
                self.kernel = K.transpose(self.tied_to.kernel)
                self._non_trainable_weights.append(self.kernel)
            else:
                self.kernel = self.add_weight(shape=(input_dim, self.units),
                                              initializer=self.kernel_initializer,
                                              name='kernel',
                                              regularizer=self.kernel_regularizer,
                                              constraint=self.kernel_constraint)
            if self.use_bias:
                self.bias = self.add_weight(shape=(self.units,),
                                            initializer=self.bias_initializer,
                                            name='bias',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
            else:
                self.bias = None
            self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
            self.built = True

        def compute_output_shape(self, input_shape):
            assert input_shape and len(input_shape) >= 2
            output_shape = list(input_shape)
            output_shape[-1] = self.units
            return tuple(output_shape)

        def call(self, inputs):
            output = K.dot(inputs, self.kernel)
            if self.use_bias:
                output = K.bias_add(output, self.bias, data_format='channels_last')
            if self.activation is not None:
                output = self.activation(output)
            return output


    class WeightsOrthogonalityConstraint (Constraint):
        def __init__(self, encoding_dim, weightage = 1.0, axis = 0):
            self.encoding_dim = encoding_dim
            self.weightage = weightage
            self.axis = axis

        def weights_orthogonality(self, w):
            if(self.axis==1):
                w = K.transpose(w)
            if(self.encoding_dim > 1):
                m = K.dot(K.transpose(w), w) - K.eye(self.encoding_dim)
                return self.weightage * K.sqrt(K.sum(K.square(m)))
            else:
                m = K.sum(w ** 2) - 1.
                return m

        def __call__(self, w):
            return self.weights_orthogonality(w)


    class UncorrelatedFeaturesConstraint (Constraint):

        def __init__(self, encoding_dim, weightage=1.0):
            self.encoding_dim = encoding_dim
            self.weightage = weightage

        def get_covariance(self, x):
            x_centered_list = []

            for i in range(self.encoding_dim):
                x_centered_list.append(x[:, i] - K.mean(x[:, i]))

            x_centered = tf.stack(x_centered_list)
            covariance = K.dot(x_centered, K.transpose(x_centered)) / \
                tf.cast(x_centered.get_shape()[0], tf.float32)

            return covariance

        # Constraint penalty
        def uncorrelated_feature(self, x):
            if(self.encoding_dim <= 1):
                return 0.0
            else:
                output = K.sum(K.square(
                    self.covariance - tf.math.multiply(self.covariance, K.eye(self.encoding_dim))))
                return output

        def __call__(self, x):
            self.covariance = self.get_covariance(x)
            return self.weightage * self.uncorrelated_feature(x)


    Tencoder = Dense(encoding_dim, activation="linear", input_shape=(input_dim,), use_bias = True, kernel_regularizer=WeightsOrthogonalityConstraint(encoding_dim, weightage=1., axis=0), kernel_constraint=UnitNorm(axis=0)) 
    Tdecoder = DenseTied(input_dim, activation="linear",  use_bias = False) #tied_to=encoder,

    Tautoencoder = Sequential()
    Tautoencoder.add(Tencoder)
    Tautoencoder.add(Tdecoder)

    return Tautoencoder
    
    
def RTPv2trainedmodel(tech_params):
    
    import tensorflow as tf
    from keras.models import Model, load_model
    from keras.layers import Input, Dense, Layer, InputSpec
    from keras.callbacks import ModelCheckpoint, TensorBoard
    from keras import regularizers, activations, initializers, constraints, Sequential
    from keras import backend as K
    from keras.constraints import UnitNorm, Constraint
    
    
    #from RTP_V2_Model_Scaffold import RTPv2model
    from pandas import read_csv as rc
    import numpy as np
    z000xfaexl1wbs0 = rc('000xfaexl1wbs[0].csv',header=None)#
    z000xfaexl1wbs1 = np.array(rc('000xfaexl1wbs[1].csv',header=None)).reshape(24,)
    z000xfaexl2wbs0 = rc('000xfaexl2wbs[0].csv',header=None)
    RTPv2modeli = RTPv2model(tech_params)

    l1wbs=[z000xfaexl1wbs0,z000xfaexl1wbs1]
    l2wbs=[z000xfaexl2wbs0]
    RTPv2modeli.layers[0].set_weights(l1wbs)
    RTPv2modeli.layers[1].set_weights(l2wbs)
    
    return RTPv2modeli
    
   
    

def scaler_module(col_list,vin_pathlist):
    try:
        from sklearn.preprocessing import MinMaxScaler
        from pandas import read_csv as rc
        from pandas import concat
        scaler = MinMaxScaler()
        #X_=rc('X_train.csv')
        X_=[]
        for i in vin_pathlist[:]:
            X_.append(rc(i))
        X= concat(X_, axis=0)
        scaler.fit(X[col_list])
        return scaler
    except:
        return 0


    
params = {
    'rtp_model':'relai_v2',
    'bucket':"relai.poc.data",
    'folder_base':'/unified_Fx_D3_power_annotated/',
    'file_type':'.csv',
    'tech_params': {
                    'encoding_dim':24,
                    'input_dim' : '', #len(get_col_module(arg1=params['rtp_model'])),
                    'batch_size' : 200,
                    'epochs':100,
                    'batch_size':200,
                    'shuffle':True    
                    },
    
    'Notes':'This is for RelAi V2 Autoencoder'
}




def entrypoint(params):
    
    print(params['Notes'])
    
    #params
    params['tech_params']['input_dim'] = len(get_col_module(arg1=params['rtp_model']))
    
    col_list = get_col_module(arg1=params['rtp_model'])
    if col_list==0: print('col_list arg not matching')
        
    vin_pathlist = input_data_module_s3(bucket =params['bucket'], folder_base = params['folder_base'],file_type =params['file_type'] )
    if vin_pathlist==0: print('vin_pathlist arg not matching')
        
    scaler=scaler_module(col_list,vin_pathlist):
    if scaler==0: print('scaler_module not initialised')
        
    testmodel1 = RTPv2trainedmodel(params['tech_params'])
    if testmodel1==0: print('testmodel1 arg not matching')
        
    return 0
    
    
    
# testrun
# entrypoint(params)
