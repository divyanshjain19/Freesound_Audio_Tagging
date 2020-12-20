import numpy as np
import utils

class Output:
    def __init__(self):
        self.weights_path = r'F:\Jupyter_Notebook\AAIC\Assignments\Case_Study_2_Freesound_Audio_Tagging\Final\best.h5'
        self.model = utils.get_model(self.weights_path)

    def predict(self, datapoint):
        '''
        Returns the final model prediction, that is, the top 3 most probable class labels along with their
        corresponding probabilities present in the clip
        Input ->
            datapoint : Path to the input .wav file
        Output ->
            results: Dictionary of size 3 containing keys as the class labels and values as their probabilities
        '''
        # Get the log mel spectrogram features for the input datapoint
        features = utils.preprocess(datapoint)

        # Take sigmoid of the predictions to get probabilitstic values between 0 and
        y_pred = utils.sigmoid(self.model.predict(features))

        # Average the predictions across all windows to get the final
        y_pred = np.average(y_pred, axis=0)

        # Get the class labels (in integer format) corresponding to the classes having the 3 highest probabilities
        top_3_prob_indices = np.argsort(y_pred)[-3:][::-1]

        # Obtain the class_map, i.e., the hash table which will be used to convert the integer class labels to English class labels
        class_map = utils.create_class_map(train_csv_path = r'F:\Jupyter_Notebook\AAIC\Assignments\Case_Study_2_Freesound_Audio_Tagging\Final\train_curated.csv')

        # Obtain the English class labels
        top_3_classes = [class_map[i] for i in top_3_prob_indices]

        # Obtain the corresponding probabilities
        top_3_prob = y_pred[top_3_prob_indices]

        # Return the result in a dictionary format
        result = {top_3_classes[i]:round(top_3_prob[i], 3) for i in range(3)}

        return result
