#
# from keras.models import Sequential
# from keras.layers import Dense
#
# from .neuralnetwork import NeuralNetwork
#
#
# class DenseRNA(NeuralNetwork):
#
#     def __init__(self, look_back=12, dense=True,
#                  lstm_cells=100, input_shape=1):
#         """Nani."""
#         self.look_back = look_back
#         self.dense = dense
#         self.lstm_cells = lstm_cells
#         self.input_shape = input_shape
#         super(DenseRNA, self).__init__()
#
#     def _create_model(self):
#         model = Sequential()
#         model.add(RNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False))
#
#
#         model.add(Dense(12, input_dim=10, init='uniform', activation='relu'))
#         model.add(Dense(10, init='uniform', activation='relu'))
#         model.add(Dense(1, init='uniform', activation='relu'))
#         model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#         # Train model
#         history = model.fit(rescaledX, Y, nb_epoch=100, batch_size=50,  verbose=1)
#         # Print Accuracy
#         scores = model.evaluate(rescaledX, Y)
#         print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#
#
# #
# # from keras.models import Sequential
# # from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
# #
# # model = Sequential()
# #
# # # Embedding layer
# # model.add(
# #     Embedding(input_dim=num_words,
# #               input_length = training_length,
# #               output_dim=100,
# #               weights=[embedding_matrix],
# #               trainable=False,
# #               mask_zero=True))
# #
# # # Masking layer for pre-trained embeddings
# # model.add(Masking(mask_value=0.0))
# #
# # # Recurrent layer
# # model.add(LSTM(64, return_sequences=False,
# #                dropout=0.1, recurrent_dropout=0.1))
# #
# # # Fully connected layer
# # model.add(Dense(64, activation='relu'))
# #
# # # Dropout for regularization
# # model.add(Dropout(0.5))
# #
# # # Output layer
# # model.add(Dense(num_words, activation='softmax'))
# #
# # # Compile the model
# # model.compile(
# #     optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
