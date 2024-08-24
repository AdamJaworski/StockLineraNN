from pathlib import Path


class PathManager():
    def __init__(self, model_name):
        self.model_path  = Path(rf'./models/{model_name}').resolve()
        self.data = Path(r'Data').resolve()
        self.out_path   = self.model_path / "output"
        self.state_path   = self.model_path / "state"
        self.settings_file = self.model_path / "settings.json"
        self.loss_file = self.model_path / "loss.txt"
        self.model_file = self.model_path / "model.py"
        self.data_train = self.data / 'DataTraining'
        self.create_dirs()

    def create_dirs(self):
        self.model_path.mkdir(exist_ok=True, parents=True)
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.state_path.mkdir(exist_ok=True, parents=True)
