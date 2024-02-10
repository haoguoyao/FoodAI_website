class Model:
    """This manages the Torch model for a specific food type
    """
    def __init__(self, name, generator=None, ingredients_vocab=None, frequences_list=None):
        """init function

        Arguments:
            name {str} -- food type name

        Keyword Arguments:
            generator {torch.nn.Module} -- Torch neural network model (default: {None})
            ingredients_vocab {list} -- list of ingredient names (default: {None})
            frequences_list {list} -- list of numbers recording the frequencies of each ingredients normalized to 1 (default: {None})
        """
        super().__init__()
        self.name = name
        self.generator = generator
        self.ingredients_vocab = ingredients_vocab
        self.frequences_list = frequences_list