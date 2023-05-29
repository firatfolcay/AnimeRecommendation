# -*- coding: utf-8 -*-
"""
Created on Mon May 29 20:09:07 2023

@author: firat
"""

import random
from math import log


class CombinedValueCalculator:
    def __init__(self):
        self.genreArray = []


# This function calculates a combined stochastic value based on ratings and popularity of a dataframe.
# Inputs:
# - dataframe: the input dataframe containing ratings and popularity information
# - closestpoints: indices of the closest points in the dataframe
# - ws: weight factor for the score
# - wp: weight factor for the popularity
# Output:
# - chosen_element: a randomly chosen element from the closest points based on the combined stochastic value
def calculate_combined_stochastic(dataframe, closestpoints, ws, wp):
    score = dataframe.loc[closestpoints, 'rating']
    popularity = dataframe.loc[closestpoints, 'members']
    combined_value = (score / 10 * ws) + (log(log(popularity.iloc[0])) * wp)
    combined_value_list = combined_value.tolist()
    chosen_element = stochastic_choose(combined_value.index, combined_value_list)
    return chosen_element


# This function performs a stochastic selection based on calculated values and returns unique chosen elements.
# Inputs:
# - list_of_chosen: list of indices to choose from
# - calculated_values: list of calculated values corresponding to each index
# Output:
# - chosen_elements_unique: a list of unique chosen elements based on stochastic selection
def stochastic_choose(list_of_chosen, calculated_values):
    total = sum(calculated_values)
    probabilities = [value / total for value in calculated_values]
    num_choices = len(list_of_chosen)
    chosen_elements = random.choices(list_of_chosen, weights=probabilities, k=num_choices)
    chosen_elements_unique = list(set(chosen_elements))
    return chosen_elements_unique
