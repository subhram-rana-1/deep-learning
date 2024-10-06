from pandas import read_csv, DataFrame
from datasets import file_paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.src.models import Sequential
from keras.src.layers import Dense
from datasets.generators.stock_up_down_classification_data_generator import \
    previous_candle_count


def get_input_output_dataset(relative_file_path: str) -> (DataFrame, DataFrame):
    df = read_csv(relative_file_path, header=None)

    input: DataFrame = df.iloc[:, :-1]
    output: DataFrame = df.iloc[:, -1]

    return input, output


def main():
    input, output = get_input_output_dataset(file_paths.stock_up_down_classification)

    input_train, input_test, output_train, output_test = \
        train_test_split(input, output, test_size=0.2, random_state=20)

    input_scaler = StandardScaler().fit(input)

    normalised_input_train = input_scaler.transform(input_train)

    ANN = Sequential()

    # ----- nonlinear feed-forward ANN model -----
    ANN.add(Dense(50, input_dim=3*previous_candle_count, activation='relu', name='hidden_1'))
    ANN.add(Dense(50, activation='relu', name='hidden_2'))
    ANN.add(Dense(50, activation='relu', name='hidden_3'))
    ANN.add(Dense(50, activation='relu', name='hidden_4'))
    ANN.add(Dense(50, activation='relu', name='hidden_5'))
    ANN.add(Dense(1, activation='linear', name='output_layer'))  # TODO
    # TODO
    # TODO : yet dont know how rto solve this using classification
    # TODO

    ANN.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    ANN.summary()

    ANN.fit(normalised_input_train, output_train, validation_split=0.2, epochs=100)

    # ---- post training evaluation on test dataset ----------
    normalised_input_test = input_scaler.transform(input_test)
    mean_squared_error, mean_absolute_error = ANN.evaluate(normalised_input_test, output_test)
    print(f'mean_squared_error: {mean_squared_error}')
    print(f'mean_absolute_error: {mean_absolute_error}')
    # --------------------------------------------------------

    # -------------- example predictions --------------- #
    normalised_input_test = input_scaler.transform(input_test)
    print(f'predicted value: \n{ANN.predict(normalised_input_test[10:30])}')
    print(f'real outputs: \n{output_test[10:30]}')
    # ------------------------------------------------ #


if __name__ == '__main__':
    main()

# Result on test data set
# -1.1841637 , -1
# -0.33600563, 0
#  0.94631904, 1
#  0.14281173, 1 ==
#  0.09401143, 0
# -1.1418205 , -1
#  0.88147527, -1 ==
#  0.6043622 , 1 == ok ok
# -0.20510808, 0
#  0.03554581, 1 ==
# -0.77721435, -1 == ok ok
#  1.0510265 , -1 ==
#  0.10282201, 1 ==
# -1.0473381 , 1 ==
# -1.5156851 , 1 ==
#  0.61479515, -1 ==
# -1.0631871 , 1 ==
#  0.8856703 , 1
#  0.9929754 , -1 ==
#  0.67284936, 1 == ok ok


# TODO ---------
# This is pretty much bad performance. This can be imporved by,
# 1. decreasing the previosu candle count
# 2. multi class slassidication