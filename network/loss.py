import numpy as np

class Loss:
    def calculate(self, y_pred, y):
        sample_loss = self.forward(y_pred, y)
        total_loss = np.mean(sample_loss)
        return total_loss
    

class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        if len(y_true) != len(y_pred):
            raise IndexError(f"Shape Error in inputs. y_pred shape: {y_pred.shape} and y_true shape: {y_true.shape}")
        
        y_true = np.array(y_true)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            pred_confidence = y_pred[range(len(y_pred)), [y_true]]
        else:
            pred_confidence = y_pred[range(len(y_true)), np.max(y_true, axis=1)]
        
        neglogloss = -np.log(pred_confidence)
        return neglogloss