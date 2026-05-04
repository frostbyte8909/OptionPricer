class OptionPricerException(Exception):
    pass

class NegativeVarianceError(OptionPricerException):
    def __init__(self, message="Variance cannot be negative"):
        self.message = message
        super().__init__(self.message)

class ConvergenceError(OptionPricerException):
    def __init__(self, message="Numerical method failed to converge"):
        self.message = message
        super().__init__(self.message)

class InvalidInputError(OptionPricerException):
    def __init__(self, message="Invalid input parameters provided"):
        self.message = message
        super().__init__(self.message)
