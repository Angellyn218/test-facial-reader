# Testing Facial Reader
This Repo has both a facial reader model and an evaluation. 

## Facial Reader
The facial reader model is Deepface with a video capture which allows the model to take a video feed directly from the user's webcam. This runs when the facial_reader.py file is ran.

## Evaluation
Using sklearn and previously labeled photo data, deepseek is tested to check its effectiveness evaluating specific emotion versus general positive/negative emotions. 

Deepface is more effective at evaluating general positive/negative emotions rather than specific emotions. The accuracy, precision, recall, and f1-scores are evaluated when evaluate.py is ran.