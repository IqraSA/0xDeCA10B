from skmultiflow.trees import HAT, RegressionHAT

from decai.simulation.contract.classification.scikit_classifier import SciKitClassifierModule


class DecisionTreeModule(SciKitClassifierModule):
    def __init__(self, regression=False):
        model_initializer = (lambda: RegressionHAT(
                # leaf_prediction='mc'
            )) if regression else (lambda: HAT(
                # leaf_prediction='mc',
                # nominal_attributes=[ 4],
            ))
        super().__init__(_model_initializer=model_initializer)
