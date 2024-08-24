import os
from settings.test_options import args
from PathManager import PathManager
import shutil

def main():
    if not args.correlation and not args.graph and not args.states:
        raise RuntimeError("You need to select test type")

    model_path_manager = PathManager(args.model)

    if os.path.exists(model_path_manager.model_file):
        os.remove('./model/model.py')
        shutil.copy(model_path_manager.model_file, './model/model.py')
    else:
        UserWarning("Didn't locate model.py in model files")

    state_name = ""
    if args.epoch != -1:
        for state in os.listdir(model_path_manager.state_path):
            if f"EOE_{args.epoch}_" in state:
                state_name = state
                break
    else:
        state_name = "latest.pth"

    if state_name == "":
        raise RuntimeError("Didn't find state from selected epoch")

    if args.correlation:
        print(f"Launching correlation test for model: {args.model} with state_dict: {state_name}")
        from test_module.test_correlation import test_correlation
        test_correlation(model_path_manager, state_name)

    if args.graph:
        assert args.data != "", RuntimeError("You need to provide data file name for graph test")

        print(f"Launching correlation test for model: {args.model} with state_dict: {state_name}")
        from test_module.test_graph import test_graph
        test_graph(model_path_manager, state_name, args.data)

    if args.states:
        assert args.data != "", RuntimeError("You need to provide data file name for states test")
        if args.epoch != -1:
            RuntimeWarning("Don't provide epoch for state test")

        print(f"Launching states test for model: {args.model} with data: {args.data}")
        from  test_module.test_states import test_states
        test_states(model_path_manager, args.data)


if __name__ == "__main__":
    main()
