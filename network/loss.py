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
    
    def derivative_self(self, output, y_true):
        n_samples = len(output)
        n_labels = len(output[0])
        
        if len(y_true.shape) == 1:
            y_true = np.eye(n_labels)[y_true]
        
        d_inputs = - y_true / output
        return d_inputs / n_samples
    
    def backward(self, probabilities, y_true):
        self.d_inputs = self.derivative_self(probabilities, y_true)
        return self.d_inputs