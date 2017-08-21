from keras.layers          import Lambda, Input, Dense, GRU, LSTM, RepeatVector
from keras.layers.core     import Flatten
from keras.callbacks       import LambdaCallback 
from keras.models          import Model
from keras.optimizers      import SGD, RMSprop, Adam
from keras.layers.wrappers import Bidirectional as Bi
from keras.layers.wrappers import TimeDistributed as TD
from keras.layers          import merge, multiply, concatenate, dot
from keras.regularizers    import l2
from keras.layers.core     import Reshape
from keras.layers.merge    import Concatenate, Dot
from keras.layers.normalization import BatchNormalization as BN
import keras.backend as K


inputs = Input(shape=(10, 10))
x      = Flatten()(inputs)
x      = Dense((100))(x)


