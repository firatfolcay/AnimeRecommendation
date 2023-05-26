import random
from math import log


class CombinedValueCalculator:
    def __init__(self):
        self.genreArray = []


def calculate_combined_stochastic(dataframe, closestpoints, ws, wp):
    score = dataframe.loc[closestpoints, 'rating']
    popularity = dataframe.loc[closestpoints, 'members']
    combined_value = (score / 10 * ws) + (log(log(popularity.iloc[0])) * wp)
    combined_value_list = combined_value.tolist()
    chosen_element = stochastic_choose(combined_value.index, combined_value_list)
    return chosen_element


def stochastic_choose(list_of_chosen, calculated_values):
    total = sum(calculated_values)
    probabilities = [value / total for value in calculated_values]
    num_choices = len(list_of_chosen)
    chosen_elements = random.choices(list_of_chosen, weights=probabilities, k=num_choices)
    chosen_elements_unique = list(set(chosen_elements))
    return chosen_elements_unique
