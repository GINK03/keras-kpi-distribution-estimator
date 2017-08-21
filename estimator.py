from keras.layers          import Lambda, Input, Dense, GRU, LSTM, RepeatVector
from keras.layers.core     import Flatten
from keras.callbacks       import LambdaCallback 
from keras.models          import Model
from keras.optimizers      import SGD, RMSprop, Adam
from keras.layers.wrappers import Bidirectional as Bi
from keras.layers.wrappers import TimeDistributed as TD
from keras.layers          import merge, multiply, concatenate, dot
from keras.regularizers    import l2
from keras.layers.core     import Reshape, Dropout
from keras.layers.merge    import Concatenate, Dot
from keras.layers.normalization import BatchNormalization as BN

import keras.backend as K
import numpy as np
import json

inputs = Input(shape=(2,))
x      = Dense(100, activation='relu')(inputs)
x      = Dropout(0.5)(x)
x      = Dense(100, activation='relu')(x)
x      = Dropout(0.5)(x)
x      = Dense(1, activation='linear')(x)
est    = Model(inputs, x)
est.compile(optimizer=Adam(), loss='mse')


# categorical to linear
SW1 = {
  '0～11歳'   :0.0, 
  '12～14歳'  :0.1,
  '15～17歳'  :0.2,
  '18～19歳'  :0.3,
  '20～21歳'  :0.4,
  '22～29歳'  :0.5,
  '30～34歳'  :0.6,
  '35～39歳'  :0.7,
  '40～49歳'  :0.8,
  '50～59歳'  :0.9,
  '60～69歳'  :1.0,
  '70歳～'    :1.1,
  }
SW2 = {
  '300万未満'   :0.0,
  '300～500万'  :0.2,
  '500～700万'  :0.4,
  '700～1000万' :0.6,
  '1000～1400万':0.8,
  '1400万以上'  :1.0
  }


# load json from KPI map
# and, normalize it
obj = json.loads( open('domain/www.buyma.com.json').read() )

## search max
m = 0.0
for age, income_freq in obj.items():
  for income, freq in income_freq.items():
    #print(age, income, freq)
    m = max(m, freq) 
print(m)

## normalize and categorical to linear
xs = []
ys = []
for age, income_freq in obj.items():
  for income, freq in income_freq.items():
    x1 = SW1[age]
    x2 = SW2[income]
    y = freq/m
    x = [x1,x2]
    print(x, y)
    xs.append( x )
    ys.append( y )

xs = np.array(xs)
ys = np.array(ys)

est.fit(xs, ys, validation_split=0.2, epochs=5000)
