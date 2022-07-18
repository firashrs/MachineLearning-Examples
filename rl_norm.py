import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
import tensorflow as tf

#Constants
R = 2
L = 5
#Variables
h = 1e-4
PI = 3.14
N = 500
Em = 220*math.sqrt(2)
f = 50

me = 311.0
mi = 0.39

plt.figure()
t = np.arange(N)
e = np.zeros(N)
for k in range(N):
    tmp_t = k*h
    e[k] = Em * math.sin(2*PI*f*tmp_t)
    #e[k] = e[k] / me
i = np.zeros(N)


i[0] = 0
for k in range(1, N):
    i[k] = ( i[k-1] * (1-((h*R)/L)) ) + ( (h/L) * e[k-1] )
    #i[k] = ( (i[k-1]/mi) * (1-((h*R)/L)) ) + ( ((h*me)/(L*mi)) * (e[k-1]) )
    #i[k] = i[k] * mi
e = e/me    
i = i/mi

plt.plot(t, e, label='e')
plt.legend()
plt.show()

plt.plot(t[:-1], i[1:], label='i', color='b')
plt.legend()
plt.show()




features = []
labels = []
for k in range(N-1):
    tmp = []
    tmp.append(i[k])
    tmp.append(e[k])
    
    features.append(tmp)
    labels.append(i[k+1])

features = np.array(features)
labels = np.array(labels)

print(features.shape)
print(labels.shape)


print('Defyning Hyperparameters')
learning_rate = 0.001
epochs = 500

print('Creating The Model')
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=(2,)))
#model.add(tf.keras.layers.Dense(units=1, input_shape=(1,2)))



model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])




model.summary()

print('Training The Model')
history = model.fit(x=features,
                    y=labels,
                    epochs=epochs)

    
print('Visualising The Loss Curve')
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('rms')
epochsss = history.epoch
hist = pd.DataFrame(history.history)
rmse = hist["root_mean_squared_error"]
plt.plot(epochsss, rmse, label='Loss')
plt.legend()
plt.ylim([rmse.min()*0.97, rmse.max()])
plt.show()



ii = model.predict_on_batch(features)

plt.plot(t[:-1], i[1:], label='i', color='b')
plt.plot(t[:250], ii[:250], label='ii', color='r')
plt.legend()
plt.show()

model.summary()

def get_predicted_R_L(weights):
    weight_1 = weights[0][0][0]
    weight_2 = weights[0][1][0]
    bias = model.get_weights()[1][0]
    L = h / weight_2
    R = ((1-weight_1)*L)/h
    
    return round(weight_1, 6), round(weight_2, 6)


ww1, ww2 = get_predicted_R_L(model.get_weights())
w1 = 1-((R/L)*h)
w2 = (h*me)/(L*mi)
print('---weights---')
print('predict=', (round(ww1,6), round(ww2, 6)))
print('orig=', (round(w1,6), round(w2, 6)))
#ww1 = round(ww1, 5)
#LL = h / ww2
#RR = ((1-ww1)*L)/h
LL = (h*me)/(ww2*mi)
RR = ((1-ww1)*LL)/h

print('---R L ---')
print('predict=', (round(LL,6), round(RR, 6)))
print('orig=', (round(L,6), round(R, 6)))
#print("__R___%.5f"%RR)
#print("__L___%.5f"%LL)
