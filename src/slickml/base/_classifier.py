from typing import Protocol


# TODO(amir): complete this; it will be prolly needed for `pipelines`
class Classifier(Protocol):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    Protocol : _type_
        _description_
    """

    def fit(self, X_train, y_train) -> None:
        """_summary_

        _extended_summary_

        Parameters
        ----------
        X_train : _type_
            _description_
        y_train : _type_
            _description_
        """
        ...
