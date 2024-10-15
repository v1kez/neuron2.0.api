import tensorflow as tf

import numpy as np

X_class = np.array([[20, 0, 0],
  [25, 1, 1],
  [30, 0, 1],
  [19, 1, 0],
# [20, 0, 0],
#   [25, 1, 1],
#   [17, 1, 0],
#   [28, 1, 1],
#   [19, 1, 0],
#   [30, 0, 1],
#   [31, 1, 1],
#   [18, 1, 1],
  ])
y_class = np.array([0,1,1,0])

# Создание модели для классификации
model_class = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_shape=(3,)),
    tf.keras.layers.Dense(3),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Один выход для бинарной классификации
])

model_class.compile(optimizer='adam', loss='binary_crossentropy')

# Обучение модели
model_class.fit(X_class, y_class, epochs=1000,verbose=False)

# Прогноз
test_data = np.array([[30, 1,1]])
y_pred_class = model_class.predict(test_data)
print("Предсказанные значения:", y_pred_class, *np.where(y_pred_class >= 0.5, 'успех', 'провал'))

test_data = np.array([[18,0,0]])
y_pred_class = model_class.predict(test_data)
print("Предсказанные значения:", y_pred_class, *np.where(y_pred_class >= 0.5, 'успех', 'провал'))
# Сохранение модели для классификации
model_class.save('classification_model.h5')