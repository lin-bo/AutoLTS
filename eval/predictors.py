import numpy as np
import pandas as pd
import torch


def lts_prediction(df):
    # initialize
    lane_div = 2 - df['oneway']
    if df['Trail'] == 1 or df['Cycle Tracks'] == 1 or df['Multi-use Pathway'] == 1:
        return 1
    if df['Bike Lanes'] == 1:
        if df['parking_indi'] == 1:
            if df['nlanes'] / lane_div <= 1:
                if df['speed_actual'] <= 40:
                    return 1
                elif df['speed_actual'] <= 48:
                    return 2
                elif df['speed_actual'] <= 56:
                    return 3
                else:
                    return 4
            else:
                if df['speed_actual'] <= 56:
                    return 3
                else:
                    return 4
        else:
            if df['nlanes'] / lane_div <= 1:
                if df['speed_actual'] <= 48:
                    return 1
                elif df['speed_actual'] <= 56:
                    return 3
                else:
                    return 4
            elif df['nlanes'] / lane_div <= 2:
                if df['speed_actual'] <= 48:
                    return 2
                elif df['speed_actual'] <= 56:
                    return 3
                else:
                    return 4
            else:
                if df['speed_actual'] <= 56:
                    return 3
                else:
                    return 4
    else:
        if df['nlanes'] <= 3:
            if df['speed_actual'] <= 40:
                if df['volume'] <= 3000 and df['Arterial'] != 1:
                    return 1
                else:
                    return 2
            elif df['speed_actual'] <= 56:
                if df['volume'] <= 3000 and df['Arterial'] != 1:
                    return 2
                else:
                    return 3
            else:
                return 4
        elif df['nlanes'] <= 5:
            if df['speed_actual'] <= 40:
                return 3
            else:
                return 4
        else:
            return 4


def lts_prediction_wo_volume(df):
    # initialize
    lane_div = 2 - df['oneway']
    if df['Trail'] == 1 or df['Cycle Tracks'] == 1 or df['Multi-use Pathway'] == 1:
        return 1
    if df['Bike Lanes'] == 1:
        if df['parking_indi'] == 1:
            if df['nlanes'] / lane_div <= 1:
                if df['speed_actual'] <= 40:
                    return 1
                elif df['speed_actual'] <= 48:
                    return 2
                elif df['speed_actual'] <= 56:
                    return 3
                else:
                    return 4
            else:
                if df['speed_actual'] <= 56:
                    return 3
                else:
                    return 4
        else:
            if df['nlanes'] / lane_div <= 1:
                if df['speed_actual'] <= 48:
                    return 1
                elif df['speed_actual'] <= 56:
                    return 3
                else:
                    return 4
            elif df['nlanes'] / lane_div <= 2:
                if df['speed_actual'] <= 48:
                    return 2
                elif df['speed_actual'] <= 56:
                    return 3
                else:
                    return 4
            else:
                if df['speed_actual'] <= 56:
                    return 3
                else:
                    return 4
    else:
        if df['nlanes'] <= 3:
            if df['speed_actual'] <= 40:
                if df['Arterial'] != 1:
                    return 1
                else:
                    return 2
            elif df['speed_actual'] <= 50:
                if df['Arterial'] != 1:
                    return 2
                else:
                    return 3
            else:
                return 4
        elif df['nlanes'] <= 5:
            if df['speed_actual'] <= 40:
                return 3
            else:
                return 4
        else:
            return 4
