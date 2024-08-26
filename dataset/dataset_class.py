class Dataset:
    """
    input_dict: dict of items for model
    gt_dict: dict of correct answers
    size: number of items in dataset
    total_size: OPTIONAL - size of dataset in KB
    """
    input_dict: list
    gt_dict: list
    size: int
    total_size: float

    def get(self, index):
        return self.input_dict[index], self.gt_dict[index]

    def get_size(self):
        return self.size