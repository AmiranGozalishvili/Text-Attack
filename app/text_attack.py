import nltk
import numpy as np
import torch
from textattack.datasets import Dataset
from textattack.models.wrappers import ModelWrapper

nltk.download('omw-1.4')
from textattack.goal_functions.classification import UntargetedClassification
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.transformations import WordSwapEmbedding
from textattack.search_methods import GreedyWordSwapWIR
from textattack import Attack
from textattack import Attacker
from textattack import AttackArgs
from keras_preprocessing.sequence import pad_sequences
from Train_Test import random_state

from data import load_data
from lstm_network import lstm_net
from tokenization import tokenize

dataset = load_data()
model = lstm_net()

maxlen, tokenizer, X_test_pad, X_train_pad, word_index = tokenize()



class CustomTensorFlowModelWrapper(ModelWrapper):
    """
    Implementation of a model wrapper class to
    run TextAttack with a custom TensorFlow model.
    """

    def __init__(self, model):
        self.model = model
        # self.tokenizer = tokenizer
        # self.maxlen = maxlen
        # self.dataset = dataset


    def __call__(self, text_input_list):
        # retrieve model prediction
        text_array = np.array(text_input_list)
        tokens = tokenizer.texts_to_sequences(text_input_list)
        tokens_pad = pad_sequences(tokens, maxlen=maxlen)
        model_pred = self.model.predict(tokens_pad)

        # return prediction scores as torch.Tensors
        logits = torch.FloatTensor(model_pred)
        logits = logits.squeeze(dim=-1)

        # for each output, index 0 corresponds to the negative
        # and index 1 corresponds to the positive confidence
        final_preds = torch.stack((1 - logits, logits), dim=1)

        return final_preds


    """## Creating the Attack"""

def create_attack():
    '''
    text attack
    '''

    # example output
    CustomTensorFlowModelWrapper(model)(["this is negative text. bad terrible awful.",
                                         "this is positive text. great amazing love"])

    # example of a successful text atack which fools the model into predicting the wrong label
    t1 = 'i love the tie dye and the accent stitching. back detail is fun!'
    t2 = 'i adore the tie colouring and the accent stitching. back detail is amusing!'
    CustomTensorFlowModelWrapper(model)([t1, t2])

    # initialize the model wrapper with the trained LSTM
    model_wrapper = CustomTensorFlowModelWrapper(model)

    # textattack requires custom datasets to be presented as a list of (input, ground-truth label) pairs
    data_pairs = []
    for input, label in zip(dataset['Review Text'], dataset['Recommended IND']):
        data_pairs.append((input, label))

    new_dataset = Dataset(data_pairs, shuffle=True)

    goal_function = UntargetedClassification(model_wrapper)

    constraints = [
        RepeatModification(),
        StopwordModification(),
        WordEmbeddingDistance(min_cos_sim=0.9)
    ]

    transformation = WordSwapEmbedding(max_candidates=50)

    search_method = GreedyWordSwapWIR(wir_method="delete")

    # construct the actual attack++
    attack = Attack(goal_function, constraints, transformation, search_method)


    # def attack_until_res():
    # attack until 1000 successfull attacks are reached
    attack_args = AttackArgs(num_examples=10,
                             random_seed=random_state)

    attacker = Attacker(attack, new_dataset, attack_args)

    attack_results = attacker.attack_dataset()
    print(attack_results)
    return attack_results

# create_attack()






# # display the attack results and the differences
# logger = CSVLogger(color_method='html')

# for result in attack_results:
#     logger.log_attack_result(result)

# from IPython.core.display import display, HTML
# display(HTML(logger.df[['original_text', 'perturbed_text']].to_html(escape=False)))
