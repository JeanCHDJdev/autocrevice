# autocrevice

----------------

**autocrevice** is a simple autoencoder setup aimed and trained to detect crevices in antartica, tested on SAR satellite imagery from the european Sentinel (ESA) database.

One week school project on the theme "Observing earth for cryosphere evolution".  
Possible improvements : more training data, testing on non-Antartica (Greenland?) glaciers and mountain glaciers
(this might need new models or new training).

Extension of this work can be found [here](https://github.com/Nicolas-Quiot/AICrevassedetection)  

**Note :** directories should be changed according to your computer and to file names, as this project was not made with path resolving in mind.  

Please install `keras`, `tensorflow`, `matplotlib`, `numpy` on installation. I recommend `model_evaluation.py` as the most complete evaluation file.

An example of the obtained results :

<img width="530" alt="Results" src="https://user-images.githubusercontent.com/119788354/234788943-a017f6a0-e659-4248-af88-252c7bf332cf.png">

Evaluation of said results using scores and the threshold for pixel detection :

<img width="530" alt="Evaluate_reply" src="https://user-images.githubusercontent.com/119788354/234789195-375d6878-f0b9-44bf-9275-d6c5e2c1b599.png">
