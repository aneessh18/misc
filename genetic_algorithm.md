### Genetic Algorithm (GA)
Neural network weights can be trained using genetic algorithms. In this case, a perceptron has been trained. Steps in the corresponding GA are
1. Create the initial perceptrons using random weights
2. Produce the offspring models from the parent models and add them to this current generation. (cross over some of the weights)
3. Mutate the weights
4. Rank the models of this generation using fitness function (no of correct labels is the metric)
5. Keep some of the best models and discard the rest.
6. Check if the expected performance has been achieved otherwise repeat the steps.

The perceptron has been able to fit AND and OR training data using this algorithm. Sometimes it cannot fit the training data because of the initial random weights or the mutation threshold. 

