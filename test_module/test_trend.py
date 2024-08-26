import os
import torch
from settings import global_variables
from dataset.trend import Trend
from settings.test_options import args

def test_trend(model_instance, data_file, state_file):
    data = Trend(global_variables.TEST_DATA_PATH / data_file)

    max_correct = 0
    best_wrong = 0
    best_state = ""
    iterate_list = os.listdir(global_variables.model_path_manager.state_path) if args.epoch == -1 else [state_file]
    for state in iterate_list:
        model_instance.load_state_dict(torch.load(global_variables.model_path_manager.state_path / state))
        correct_ = 0
        wrong_   = 0
        with torch.no_grad():
            for i in range(data.get_size()):
                input_tensor, correct_tensor = data.get(i)
                output_tensor = model_instance(input_tensor)

                if args.epoch != -1:
                    print(f"{i}. Output: {output_tensor}, Correct: {correct_tensor}")

                if torch.argmax(output_tensor) == torch.argmax(correct_tensor):
                    correct_ += 1
                else:
                    wrong_ += 1

        print(f"Instance: {state}, Correct: {correct_}, Wrong: {wrong_}")

        if correct_ > max_correct:
            max_correct = correct_
            best_wrong = wrong_
            best_state = state

    if args.epoch == -1:
        print(f"Best state: {best_state} with correct: {max_correct} and wrong: {best_wrong}")