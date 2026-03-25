import torch
import time
import math
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from collections import defaultdict
import json
import string
import numpy as np
from gensim.test.utils import common_texts, datapath
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
import ast
import random
from tqdm import tqdm

best_av_size = 100
best_params = [100, 100]

# Define the number of shots, sampling counts, alpha
kshot = None
sampling_num = 20

quantile_bar = 0.2
results = []

# Initialize the similarity model
similarity_model = FastText(sentences=common_texts, vector_size=200, min_count=1)


def new_CP_score(dict_of_freq, weight, weight_2):
    """Compute conformal prediction nonconformity scores for each response.

    Args:
        dict_of_freq: Dict mapping response text to frequency count.
        weight: Weight for the entropy term.
        weight_2: Weight for the similarity penalty term.

    Returns:
        Tuple of (dict_of_score, normalized_entropy) where dict_of_score maps
        each response to its nonconformity score.
    """
    dict_of_score = dict_of_freq.copy()
    total_frequency = sum(dict_of_freq.values())

    numerator = 0
    for key, value in dict_of_score.items():
        numerator += - value / total_frequency * math.log(value / total_frequency)
    if total_frequency == 1 or total_frequency == 0:
        total_frequency = 2
    normalized_entropy = numerator / math.log(total_frequency)

    rank_1_response = ""
    for rank, (key, value) in enumerate(dict_of_score.items()):
        if rank == 0:
            rank_1_response = key
            dict_of_score[key] = 10 - value / total_frequency * 10 + normalized_entropy / 2 * weight
        else:
            dict_of_score[key] = 10 - value / total_frequency * 10 + normalized_entropy / 2 * weight
            try:
                dict_of_score[key] -= similarity_model.wv.similarity(key, rank_1_response) * weight_2
            except KeyError:
                pass

    return dict_of_score, normalized_entropy


def calculate_quantile(n, alpha):
    result = np.ceil((n + 1) * (1 - alpha)) / n
    return result


def remove_punctuation(input_string):
    translator = str.maketrans("", "", string.punctuation)
    return input_string.translate(translator)

def remove_articles(input_string):
    articles = ['a', 'an', 'the']
    words = input_string.split()
    result = ' '.join(word for word in words if word.lower() not in articles)
    return result

def remove_duplicate_whitespace(input_string):
    return ' '.join(input_string.split())

def process_list_of_strings(input_list):
    return [remove_duplicate_whitespace(remove_articles(remove_punctuation(item.lower())))
            for item in input_list]

def process_list_of_dicts(input_list_of_dicts):
    processed_list_of_dicts = []

    for dictionary in input_list_of_dicts:
        processed_dict = {}
        for key, value in dictionary.items():
            processed_key = remove_duplicate_whitespace(remove_articles(remove_punctuation(key.lower())))

            if processed_key in processed_dict:
                processed_dict[processed_key] += value
            else:
                processed_dict[processed_key] = value

        processed_list_of_dicts.append(processed_dict)

    return processed_list_of_dicts


def apply_conformal_prediction(test_generation, best_params, quantile_value):
    """Apply conformal prediction to test set using calibrated parameters.

    Args:
        test_generation: List of frequency dicts for test samples.
        best_params: [weight, weight_2] tuple of best hyperparameters.
        quantile_value: Calibrated quantile threshold.

    Returns:
        List of prediction sets (lists of response strings).
    """
    weight, weight_2 = best_params
    predicted_answers = []

    for dict_of_freq in tqdm(test_generation):
        result, _ = new_CP_score(dict_of_freq, weight, weight_2)
        predicted_answer = {key: score for key, score in result.items() if score <= quantile_value}
        predicted_answers.append(list(predicted_answer.keys()))

    return predicted_answers


def list_of_lists_to_frequency_dicts(list_of_lists):
    '''
    Convert the list to sorted dicts showing the frequencies of generated answers
    '''
    frequency_dicts = []
    for sub_list in list_of_lists:
        element_frequency = defaultdict(int)
        for element in sub_list:
            element_frequency[element] += 1
        sorted_frequency = dict(sorted(element_frequency.items(), key=lambda item: item[1], reverse=True))
        frequency_dicts.append(sorted_frequency)
    return frequency_dicts


weights = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
weights_2 = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
best_quantilte_value = 0
for weight in weights:
    for weight_2 in weights_2:
        print(f"weight: {weight}, weight_2: {weight_2}")
        TEST_number_of_freq1 = 0
        cali_answer = []
        test_answer = []
        correct_answers = []
        generation_calibration = []
        generation_test = []
        generation = []

        # Load the accuracy-based correct answers
        with open('generation_llama2.txt', 'r', encoding='utf-8') as file:
            content = file.read()
            correct_answers = ast.literal_eval(content)

        with open('generation_llama2_accuracy.txt', 'r', encoding='utf-8') as file:
            content = file.read()
            generation = ast.literal_eval(content)

        # Process correct_answers and generation
        correct_answers = process_list_of_dicts(correct_answers)
        generation = process_list_of_dicts(generation)

        combined_data = list(zip(correct_answers, generation))

        total_correct_sum = 0
        total_size_sum = 0
        total_val_size_sum = 0
        size_distribution_sum = [0] * 31  # Assuming max size is 30
        coverage_sum = [0] * 31
        random_count = 50
        current_count = 0

        # for example, num = 3
        num = 3

        for i in range(random_count):
            random.shuffle(combined_data)
            correct_answers_shuffled, generated_answers_freq = zip(*combined_data)
            nonconformity_scores = []


            test_set = []
            test_correct_answers = []
            val_set = []
            val_correct_answers = []
            bar = 0

            for index, dict_of_freq in enumerate(generated_answers_freq):
                if index % num == 0:
                    # Extract context and accuracy
                    correct_dict = correct_answers_shuffled[index]
                    context, accuracy = next(iter(correct_dict.items()))

                    # Determine correctness based on accuracy
                    is_correct = accuracy >= 0.7

                    if is_correct:
                        if context in dict_of_freq:
                            # Correct answer is present
                            nonconformity_score = new_CP_score(dict_of_freq, weight, weight_2)[0][context]
                            if (dict_of_freq[context] == 1):
                                TEST_number_of_freq1 += 1
                        else:
                            # Correct answer not present
                            nonconformity_score = 20
                    else:
                        # Consider as incorrect
                        nonconformity_score = 20

                    nonconformity_scores.append(nonconformity_score)
                    bar += 1
                elif index % num == 1:
                    val_set.append(generated_answers_freq[index])
                    val_correct_answers.append(correct_answers_shuffled[index])
                else:
                    test_set.append(generated_answers_freq[index])
                    test_correct_answers.append(correct_answers_shuffled[index])

            quantile = calculate_quantile(bar, quantile_bar) * 100
            sorted_nonconformity_scores = sorted(nonconformity_scores, reverse=True)

            quantile_value = np.percentile(sorted_nonconformity_scores, quantile)

            current_count += 1
            predicted_answers_val = []
            predicted_answers = []

            for index, dict_of_freq in enumerate(val_set):
                result, NE = new_CP_score(dict_of_freq, weight, weight_2)
                predicted_answer = {key: score for key, score in result.items() if score <= quantile_value}
                predicted_answers_val.append(list(predicted_answer.keys()))

            for index, dict_of_freq in enumerate(test_set):
                result, NE = new_CP_score(dict_of_freq, weight, weight_2)
                predicted_answer = {key: score for key, score in result.items() if score <= quantile_value}
                predicted_answers.append(list(predicted_answer.keys()))

            total_val_size = 0
            total_size = 0
            size_distribution = [0] * 31
            conditional_coverage = [0] * 31
            total_question = 0
            total_val_question = 0
            total_correct = 0

            # Evaluate validation set
            for sublist in predicted_answers_val:
                val_sublist_len = len(sublist)
                total_val_question += 1
                total_val_size += val_sublist_len
            total_val_size_sum += total_val_size

            # Evaluate test set
            for k, sublist in enumerate(predicted_answers):
                total_question += 1
                correct_dict = test_correct_answers[k]
                context, accuracy = next(iter(correct_dict.items()))
                is_correct = accuracy >= 0.7

                if is_correct:
                    if context in sublist:
                        total_correct += 1
                        conditional_coverage[len(sublist)] += 1
                    else:
                        pass
                else:
                    pass

                total_size += len(sublist)

                if len(sublist) < len(size_distribution):
                    size_distribution[len(sublist)] += 1
                else:
                    size_distribution[-1] += 1  # Assign to the last bin if exceeds

            total_size_sum += total_size
            total_correct_sum += total_correct
            for i in range(len(size_distribution)):
                size_distribution_sum[i] += size_distribution[i]
                coverage_sum[i] += conditional_coverage[i]

        print("BEGIN===============================================================")
        print("average size ", total_size_sum / random_count / total_question)
        print("distribution: ", [x / random_count for x in size_distribution_sum])
        print("conditional cov: ", [x / random_count for x in coverage_sum])
        print("accuracy: ", total_correct_sum / total_question / random_count)
        print(total_correct_sum)
        print(total_question)
        print(random_count)
        print("OVER===============================================================")
        if total_val_size_sum / random_count / total_val_question < best_av_size:
            best_av_size = total_val_size_sum / random_count / total_val_question
            best_params[0] =  weight
            best_params[1] =  weight_2
            best_quantilte_value = quantile_value
        results.append([total_size_sum / random_count / total_question, total_correct_sum / total_question / random_count])

print(best_av_size)
print(best_params)
print(best_quantilte_value)


# Inference stage
dict_list = []
with open('generation_test_llama2.txt', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if line:
            dict_item = ast.literal_eval(line)
            dict_list.append(dict_item)

prediction_set = apply_conformal_prediction(dict_list,best_params,best_quantilte_value)

with open(f"prediction_set_quantile{quantile_bar}_threshold0.7_llama2.json", "w", encoding="utf-8") as file:
    json.dump(prediction_set, file, ensure_ascii=False, indent=4)
