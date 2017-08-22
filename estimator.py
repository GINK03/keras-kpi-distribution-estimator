from keras.layers          import Lambda, Input, Dense, GRU, LSTM, RepeatVector
from keras.layers.core     import Flatten
from keras.callbacks       import LambdaCallback 
from keras.models          import Model, load_model
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
import sys
import random
import pickle
import re

inputs = Input(shape=(2,))
x      = Dense(200, activation='relu')(inputs)
x      = Dropout(0.2)(x)
x      = Dense(200, activation='relu')(x)
x      = Dropout(0.2)(x)
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
RSW1 = { f:k for k,f in SW1.items() }
SW2 = {
  '300万未満'   :0.0,
  '300～500万'  :0.2,
  '500～700万'  :0.4,
  '700～1000万' :0.6,
  '1000～1400万':0.8,
  '1400万以上'  :1.0
  }
RSW2 = { f:k for k,f in SW2.items() }

# load json from KPI map
# and, normalize it
obj = json.loads( open('domain/www.buyma.com.json').read() )

## search max
m = 0.0
for age, income_freq in obj.items():
  for income, freq in income_freq.items():
    # print(age, income, freq)
    m = max(m, freq) 
print(m)

## normalize and categorical to linear
if '--make_dataset' in sys.argv:
  xs = []
  ys = []
  vxs = []
  vys = []
  totalx = []
  totaly = []
  for age, income_freq in obj.items():
    for income, freq in income_freq.items():
      x1 = SW1[age]
      x2 = SW2[income]
      y = freq/m
      x = [x1,x2]
      totalx.append( x )
      totaly.append( y )
      if random.random() < 0.6:
        print(x, y)
        xs.append( x )
        ys.append( y )
      else:
        vxs.append( x )
        vys.append( y )

  xs = np.array(xs)
  ys = np.array(ys)
  vxs = np.array(vxs)
  vys = np.array(vys)
  totalx = np.array( totalx )
  totaly = np.array( totaly )
  open('dataset/dataset.pkl', 'wb').write( pickle.dumps([xs, ys, vxs, vys, totalx, totaly]) )

if '--make_sample_dataset' in sys.argv:
  xs = []
  ys = []
  vxs = []
  vys = []
  totalx = []
  totaly = []
  for line in open('sample/beta.txt'):
    ents = re.split(r'\t', line.strip())
    x1, x2, y = float(ents[0]), float(ents[1]), float(ents[-1])
    x = [x1,x2]
    totalx.append( x )
    totaly.append( y )
    if random.random() < 0.8:
      print(x, y)
      xs.append( x )
      ys.append( y )
    else:
      vxs.append( x )
      vys.append( y )

  xs = np.array(xs)
  ys = np.array(ys)
  vxs = np.array(vxs)
  vys = np.array(vys)
  totalx = np.array( totalx )
  totaly = np.array( totaly )
  open('dataset/dataset.pkl', 'wb').write( pickle.dumps([xs, ys, vxs, vys, totalx, totaly]) )

if '--train' in sys.argv:
  xs, ys, vxs, vys, totalx, totaly = pickle.loads( open('dataset/dataset.pkl', 'rb').read() )
  est.fit(xs, ys, validation_data=(vxs, vys), epochs=5000)
  est.save_weights('est.h5')

if '--predict' in sys.argv:
  xs, ys, vxs, vys, totalx, totaly = pickle.loads( open('dataset/dataset.pkl', 'rb').read() )
  est.load_weights('est.h5') 
  pys = est.predict(totalx)
  #for x,y in zip(xs.tolist(), ys.tolist()):
  #  print('input', ' '.join(map(str,x)), y)
 
  #sys.exit()
  age_income_freq = {}
  for x, y in zip(totalx.tolist(), ys.tolist()):
    print( x, y )
    age = RSW1[ x[0] ]
    income = RSW2[ x[1] ]
    if age_income_freq.get(age) is None:
      age_income_freq[age] = {}
    #if age_income_freq[age].get( income ) is None:
    age_income_freq[age][income] = y.pop()
  open('predict.json', 'w').write( json.dumps(age_income_freq, indent=2, ensure_ascii=False) )   

  # original 
  age_income_freq = {}
  for x, y in zip(totalx.tolist(), totaly.tolist()):
    age = RSW1[ x[0] ]
    income = RSW2[ x[1] ]
    if age_income_freq.get(age) is None:
      age_income_freq[age] = {}
    #if age_income_freq[age].get( income ) is None:
    age_income_freq[age][income] = y
  open('origial.json', 'w').write( json.dumps(age_income_freq, indent=2, ensure_ascii=False) )   
  
  # drop
  age_income_freq = {}
  for x, y in zip(xs.tolist(), ys.tolist()):
    age = RSW1[ x[0] ]
    income = RSW2[ x[1] ]
    if age_income_freq.get(age) is None:
      age_income_freq[age] = {}
    #if age_income_freq[age].get( income ) is None:
    age_income_freq[age][income] = y.pop()
  open('drop.json', 'w').write( json.dumps(age_income_freq, indent=2, ensure_ascii=False) )   
