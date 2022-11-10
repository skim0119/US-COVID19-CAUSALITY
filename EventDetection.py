import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import odr

from functools import partial
from functools import cache

from tqdm import tqdm

# Implementation of: V. Gurainik, J. Srivastava, Event detection from time series data (1999)
# To track the equations as exactly as possible, all the variable names are kept same as the paper.

class EventDetection:

    def __init__(self, p:int=8):
        self.p = p-1  # Polynomial Degree
        self.Change_points = set()  # stored in index
        self.Candidates = set()
        self.MSet = [
            odr.polynomial(self.p)
        ]
        self.Q = {}

    def pick_likelihood_criteria(self, T):
        min_c = None
        min_likelihood = None
        for c in list(self.Candidates):
            prev_cp = 0
            likelihood = 0
            for change_point in list(self.Change_points) + [len(T)]:
                if c < change_point:
                    if (prev_cp, c) not in self.Q:
                        self.Q[(prev_cp, c)] = self.find_likelihood_criteria(T[prev_cp:c])
                    if (c, change_point) not in self.Q:
                        self.Q[(c, change_point)] = self.find_likelihood_criteria(T[c:change_point])
                    likelihood += self.Q[(prev_cp, c)] + self.Q[(c, change_point)]
                prev_cp = change_point
            if min_likelihood is None or likelihood < min_likelihood:
                min_likelihood = likelihood
                min_c = c
        return min_c, min_likelihood

    def rss(self, y, f):
        """
        residual sum of square
        RSS = sum_i^n (y_i - f(x_i))**2
        """
        return np.linalg.norm(y - f(y.index), ord=2)

    def risk(self, T, M):
        """
        Compute risk
        """
        # Fit
        n = len(T)
        t = T.index
        data = odr.Data(t, T)
        odr_obj = odr.ODR(data, M)
        output = odr_obj.run()
        poly = np.poly1d(output.beta[::-1])
        predict = poly(t)
        
        return np.linalg.norm(predict - T), poly

    def fit(self, T, m):
        """
        m: model after the regression
        """
        # Return likelihood criteria: -2log(L)
        def m_sigma_square(T):
            return len(T) * (self.rss(T, m)) ** 2
        likelihood_criteria = 0
        sindex = 0
        for idx in list(self.Change_points):
            likelihood_criteria += m_sigma_square(T[sindex:idx])
            sindex = idx
        likelihood_criteria += m_sigma_square(T[sindex:])
        return likelihood_criteria

    # Figure 3
    def find_likelihood_criteria(self, T):
        minimum_risk = None
        for M in self.MSet:
            model_risk, m = self.risk(T, M)
            if minimum_risk is None or model_risk < minimum_risk:
                minimum_risk = model_risk
                likelihood_criteria = self.fit(T, m)
        return likelihood_criteria

    # Figure 2
    def find_candidate(self, T, sindex=0, eindex=None):
        eindex = eindex or len(T)
        T = T.iloc[sindex:eindex]
        optimal_likelihood_criteria = None
        split = len(T)//2
        for i in tqdm(range(self.p, len(T) - self.p)):
            likelihood_criteria = self.find_likelihood_criteria(T[:i]) + self.find_likelihood_criteria(T[i:])
            if optimal_likelihood_criteria is None or likelihood_criteria < optimal_likelihood_criteria:
                split = sindex + i
                optimal_likelihood_criteria = likelihood_criteria
        return split

    # Procedure (Figure 1)
    def detect_change_points(self, T):
        T.index = T.index.astype(int) / (10**9)
        T.index -= T.index.min()
        new_change_point = self.find_candidate(T)

        T1_range, T2_range = self.get_new_time_ranges(T, new_change_point)
        self.Change_points.add(new_change_point)

        prev_L = 0
        stopping_criteria = False
        stopping_threshold = 0.0
        while not stopping_criteria:
            c1 = self.find_candidate(T, *T1_range)
            c2 = self.find_candidate(T, *T2_range)
            self.Candidates.add(c1)
            self.Candidates.add(c2)
            print(T1_range, T2_range)
            print(new_change_point)
            print(c1)
            print(c2)
            print("-"*10)
            input('')

            new_change_point, L = self.pick_likelihood_criteria(T)
            self.Candidates.remove(new_change_point)
            T1_range, T2_range = self.get_new_time_ranges(T, new_change_point)
            self.Change_points.add(new_change_point)

            stopping_criteria = (prev_L - L) < stopping_threshold
            prev_L = L

    def get_new_time_ranges(self, T, new_change_point):
        arr = np.concatenate([[0], list(self.Change_points), [len(T)]]).astype(np.int_)
        imax = arr[np.where(arr > new_change_point)[0][0]]
        imin = arr[np.where(arr < new_change_point)[0][-1]]
        return (imin, new_change_point), (new_change_point, imax)

def debug():
    from CovidRawDataManager import CovidRawDataManager

    covid_raw_data_manager = CovidRawDataManager()
    statewise_cases_df = covid_raw_data_manager.generate_statewise_history_data(
        states_df_label="cases"
    )
    df = statewise_cases_df.diff().rolling("7D").sum().dropna()

    #df.Alabama.plot()
    data = df.Alabama
    print(f"{data.shape=}")

    ed = EventDetection()
    print(ed.detect_change_points(data.iloc[:100]))

if __name__=='__main__':
    debug()
