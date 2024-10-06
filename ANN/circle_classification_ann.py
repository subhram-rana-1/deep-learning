from pandas import read_csv, DataFrame
from datasets import file_paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.src.models import Sequential
from keras.src.layers import Dense


def get_input_output_dataset(relative_file_path: str) -> (DataFrame, DataFrame):
    df = read_csv(relative_file_path)

    input = df[['x', 'y']]
    output = df[['outside_circles', 'c1', 'c2', 'c3']]

    print(input.head())
    print(output.head())

    return input, output


def main():
    input, output = get_input_output_dataset(file_paths.circle_classification)

    input_train, input_test, output_train, output_test = \
        train_test_split(input, output, test_size=0.2, random_state=20)

    input_scaler = StandardScaler().fit(input)

    normalised_input_train = input_scaler.transform(input_train)

    ANN = Sequential()

    # ----- linear feed-forward ANN model -----
    # ANN.add(Dense(1, input_dim=2, activation='linear', name='output_layer'))  # linear model

    # ----- nonlinear feed-forward ANN model -----
    ANN.add(Dense(50, input_dim=2, activation='relu', name='hidden_1'))
    ANN.add(Dense(50, activation='relu', name='hidden_2'))
    ANN.add(Dense(50, activation='relu', name='hidden_3'))
    ANN.add(Dense(50, activation='relu', name='hidden_4'))
    ANN.add(Dense(50, activation='relu', name='hidden_5'))
    ANN.add(Dense(4, activation='relu', name='output_layer'))

    ANN.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    ANN.summary()

    ANN.fit(normalised_input_train, output_train, validation_split=0.2, epochs=100)

    # ---- post training evaluation --
    normalised_input_test = input_scaler.transform(input_test)
    mean_squared_error, mean_absolute_error = ANN.evaluate(normalised_input_test, output_test)
    print(f'mean_squared_error: {mean_squared_error}')
    print(f'mean_absolute_error: {mean_absolute_error}')
    # --------------------------------

    # -------------- example predictions --------------- #
    normalised_input_test = input_scaler.transform(input_test)
    print(f'predicted value: \n{ANN.predict(normalised_input_test[:20])}')
    print(f'real outputs: \n{output_test[:20]}')
    # ------------------------------------------------ #


if __name__ == '__main__':
    main()
