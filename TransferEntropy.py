from typing import Optional

import os
import random
import time
import itertools

import pandas as pd
import numpy as np
from tqdm import tqdm


class TransferEntropy:
    """
    Calculates information transfer motivated by [1]_.
    Particularly, we tracked the derivation based on Shannon entropy.
    The class object can be called to compute transfer-entropy matrix
    given column-wise dataframe.

    .. code-block::py

        te = TransferEntropy()
        directional_entropy = te(df)

    .. [1] Schreiber T., Measuring information transfer, 2000

    Parameters:
    -----------
        k: int
            dimension of b, number of samples of the past of b
        l: int
            dimension of a, number of samples of the past of a
        h:
            instant in the future of b
    """

    def __init__(self, k, l, h):
        self.k = k
        self.l = l
        self.h = h

        self.save_path = "de.npz"

    def __call__(
        self, df: pd.DataFrame, threshold: Optional[float] = None, disable_progbar=False
    ):
        # if os.path.exists(self.save_path):
        #    transfer_entropy = np.load(self.save_path)["directional_entropy"]
        #    return transfer_entropy
        """Evaluate Transfer entropy"""
        num_columns = df.columns.size
        transfer_entropy = np.zeros([num_columns, num_columns])
        for i in tqdm(range(0, num_columns), position=0, disable=disable_progbar):
            for j in tqdm(
                range(i + 1, num_columns), position=1, disable=disable_progbar
            ):
                transfer_entropy[i][j] = self.compute_bivariate_transfer_entropy(
                    df[df.columns[i]], df[df.columns[j]]
                )
                transfer_entropy[j][i] = self.compute_bivariate_transfer_entropy(
                    df[df.columns[j]], df[df.columns[i]]
                )
        np.savez(
            self.save_path, directional_entropy=transfer_entropy
        )  # TODO: maybe accomodate k,l,h
        return transfer_entropy

    def compute_bivariate_transfer_entropy(self, a, b):
        """
        Compute transfer entropy between two streams a and b (a -> b)
        """
        joint_probability_ab = self.joint_probability(a, b)

        reduced_joint_prob = self.joint_prob_reduce(joint_probability_ab)
        conditional_numerator = self.conditional_prob(joint_probability_ab)
        conditional_denominator = self.conditional_prob(
            reduced_joint_prob, use_zero_l=True
        )
        conditional_prob_by_division = self.conditional_division(
            conditional_numerator, conditional_denominator
        )

        nonzero_indices = conditional_prob_by_division != 0
        log2_conditional_prob_by_cond = np.log2(
            conditional_prob_by_division[nonzero_indices]
        )
        te = np.sum(
            joint_probability_ab[nonzero_indices] * log2_conditional_prob_by_cond
        )
        return te

    def joint_probability(self, a, b):
        """
        Compute joint probability
        joint probability p(i_t+1), i_t^k, j_t^l)
        """

        N = a.shape[0]
        n_states = 2 ** (self.k + self.l + 1)
        combinations = self.combinations(self.k + self.l + 1)
        count = np.zeros(n_states)
        prob = np.zeros(n_states)
        a_prob_index = []
        b_prob_index = []
        index = max(self.k, self.l) - 1
        for i in np.arange(index, N - self.h):
            b_prob_index.extend(b[i - np.arange(self.k)])
            a_prob_index.extend(a[i - np.arange(self.l)])

            ab = [b[i + self.h]] + b_prob_index + a_prob_index  # Concatenate
            index_combination = combinations.index(ab)
            count[index_combination] += 1

            a_prob_index = []
            b_prob_index = []

        return count / sum(count)  # Probability occurence

    def conditional_prob(self, joint_prob, use_zero_l=False):
        if use_zero_l:
            l = 0
        else:
            l = self.l
        combinations = self.combinations(self.k + l + 1)
        states = self.combinations(self.k + l)

        conditional = np.zeros(2 ** (self.k + l + 1))
        for i, state in enumerate(states):
            index_zero = combinations.index([0] + state)
            prob_prior = joint_prob[index_zero]

            index_one = combinations.index([1] + state)
            prob_later = joint_prob[index_one]

            if prob_prior + prob_later != 0:
                idx_offset = 2 ** (self.k + l)
                conditional[i] = prob_prior / (prob_prior + prob_later)
                conditional[i + idx_offset] = prob_later / (prob_prior + prob_later)
        return conditional

    def joint_prob_reduce(self, joint_prob):
        """
        Joint probability evaluation p(i_t+h, i_t**k)
        """
        combinations = self.combinations(self.k + self.l + 1)
        states = self.combinations(self.k + 1)
        joint = np.zeros(2 ** (self.k + 1))

        for i, state in enumerate(states):
            for j, c in enumerate(combinations):
                if c[0 : self.k + 1] == state:
                    joint[i] = joint[i] + joint_prob[j]
        return joint

    def conditional_division(self, numerator, denominator):
        """
        Division of the conditionals in log2
        """
        combinations = self.combinations(self.k + self.l + 1)
        density = self.combinations(self.k + 1)
        div = np.zeros_like(numerator)
        for j, c in enumerate(combinations):
            if not denominator[density.index(c[0 : self.k + 1])]:
                div[j] = numerator[j] / denominator[density.index(c[0 : self.k + 1])]
        return div

    def combinations(self, length):
        bin_combination = itertools.product([0, 1], repeat=length)
        bin_combination = map(list, bin_combination)
        return list(bin_combination)
