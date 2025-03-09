### Let's try some simulations!

This folder contains the code for the simulation analysis of our models.

* Our analysis ran 1000 iterations of select episodes of our trained model.
* Since training involves randomness and is very seed-dependent, this part aimed to look at possible outcomes of the training and testing phases.
* Note that simulations of the training phase are *not* updating parameters
* You can run training simulations with `training_simulation_analysis.py` and testing simulations with `testing_simulation_analysis.py`.



Based on these analyses, we were able to observe how the AI was approaching the training and testing phases, primarily by identifying patterns among decisions made by the model.

Before running, make sure you have the proper model checkpoints loaded!