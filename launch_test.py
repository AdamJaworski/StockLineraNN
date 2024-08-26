from settings.test_options import args
from settings.common_options import common_args
from launch_funcs import *

def main():
    assert args.correlation or args.graph or args.states or args.trend, "You need to select test type"

    init_model_path_manager()
    load_model_file()

    from model.model import Model
    model = Model()

    set_device()
    common_args.load_settings = True
    load_model_settings()

    state_name = ""
    if args.epoch != -1:
        for state in os.listdir(global_variables.model_path_manager.state_path):
            if f"EOE_{args.epoch}_" in state:
                state_name = state
                break
    else:
        state_name = "latest.pth"

    if state_name == "":
        raise RuntimeError("Didn't find state from selected epoch")

    model.load_state_dict(torch.load(global_variables.model_path_manager.state_path / state_name))

    if args.correlation:
        print(f"Launching correlation test for model: {common_args.model} with state_dict: {state_name}")
        from test_module.test_correlation import test_correlation
        test_correlation(model)

    if args.graph:
        assert args.data != "", RuntimeError("You need to provide data file name for graph test")

        print(f"Launching correlation test for model: {common_args.model} with state_dict: {state_name}")
        from test_module.test_graph import test_graph
        test_graph(model, args.data)

    if args.states:
        assert args.data != "", RuntimeError("You need to provide data file name for states test")
        if args.epoch != -1:
            RuntimeWarning("Don't provide epoch for state test")

        print(f"Launching states test for model: {common_args.model} with data: {args.data}")
        from  test_module.test_states import test_states
        test_states(model, args.data)

    if args.trend:
        print(f"Launching trend test for model: {common_args.model} with state_dict: {state_name}")
        from test_module.test_trend import test_trend
        test_trend(model, args.data, state_name)



if __name__ == "__main__":
    main()
