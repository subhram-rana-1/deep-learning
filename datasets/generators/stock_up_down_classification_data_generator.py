import csv
import json
import os
from typing import List
import requests
from datasets import file_paths


NSE_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
end_date = '2024-10-04'
start_date = '2024-04-04'  # max upto 6 months
previous_candle_count = 30  # previous 30 minutes move
trade_till_next_candle_count = 6  # 6 minutes
max_candle_wait = 10  # wait max 10 min for a trade exit if entered at current candle
fixed_target = 10
fixed_loss = 5


def get_olhc_candlesticks(start_date, end_date) -> List[List[float]]:
    resp = requests.get(
        f"https://api.upstox.com/v2/historical-candle/NSE_INDEX%7CNifty%2050/1minute/"
        f"{end_date}/{start_date}"
    )

    resp_dict = json.loads(resp.content)

    if resp.status_code != 200:
        raise Exception(f'non 200 repsonse: {resp.status_code}, body: {resp_dict}')

    candles = resp_dict['data']['candles']
    candles = reversed(candles)

    olhc = []
    for candle in candles:
        olhc.append(candle[1:5])

    return olhc


def transform_OLHC_candle(olhc_candlestick: List[float]) -> List[float]:
    open, high, low, close = \
        olhc_candlestick[0], olhc_candlestick[1], olhc_candlestick[2], olhc_candlestick[3]

    body = round(close-open, 2)
    upper_wick = round(high-max(open, close), 2)
    lower_wick = round(min(open, close)-close, 2)

    return [body, upper_wick, lower_wick]


def get_move_direction(
        olhc_candlesticks: List[List[float]],
        n: int, i: int,
        max_candle_wait: int,
        target_diff: int, loss_diff: int,
) -> int:
    """direction
    1 -> up
    -1 -> down
    0 -> range bound
    """

    prev_candle_close = olhc_candlesticks[i-1][3]

    upper_move_target = prev_candle_close + target_diff
    upper_move_stoploss = prev_candle_close - loss_diff

    lower_move_target = prev_candle_close - target_diff
    lower_move_stoploss = prev_candle_close + loss_diff

    j = i
    while j < min(i+max_candle_wait, n):
        high, low = olhc_candlesticks[j][1], olhc_candlesticks[j][2]

        # check for UP move catching
        if high > upper_move_target:
            # check if SL was hit in b.w this move or not
            sl_hit = False
            k = i
            while k <= j:
                low = olhc_candlesticks[j][2]
                if low <= upper_move_stoploss:
                    sl_hit = True
                    break

                k += 1

            if not sl_hit:
                return 1

        # check for DOWN move catching
        elif low < lower_move_target:
            # check if SL was hit in b.w this move or not
            sl_hit = False
            k = i
            while k <= j:
                high = olhc_candlesticks[j][1]
                if high >= lower_move_stoploss:
                    sl_hit = True
                    break

                k += 1

            if not sl_hit:
                return -1

        j += 1

    return 0


def main():
    csv_file = os.path.abspath(file_paths.stock_up_down_classification)
    if os.path.exists(csv_file):
        os.remove(csv_file)

    olhc_candlesticks = get_olhc_candlesticks(start_date, end_date)

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        column_names = []
        for i in range(previous_candle_count, 0, -1):
            column_names.append(f'candle_{i}')
        column_names.append('move_direction')
        writer.writerow(column_names)

        n = len(olhc_candlesticks)

        previous_candle_info = []
        i = 0
        while i < n:
            # 1 do prediction is possible
            if len(previous_candle_info) == previous_candle_count:
                move_direction = get_move_direction(olhc_candlesticks, n, i, max_candle_wait,
                                                    fixed_target, fixed_loss)

                # write the row to csv file
                row = previous_candle_info.copy()
                row.append(move_direction)

                writer.writerow(row)

            # 2. add the current candle
            previous_candle_info.append(transform_OLHC_candle(olhc_candlesticks[i]))

            # 3. remove rom the starting of the candle series
            if len(previous_candle_info) > previous_candle_count:
                previous_candle_info.pop(0)

            i += 1


if __name__ == '__main__':
    main()
