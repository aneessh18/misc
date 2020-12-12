import random 
import numpy as np

class Perceptron:
    def __init__(self, no_of_features, learning_rate, epochs):  #initializing the model parameters required for training
        self.no_of_features = no_of_features
        self.epochs = epochs
        self.learning_rate = learning_rate #bias is added as weight
        self.weights = np.zeros(no_of_features+1) # dimensions of weight vector that is created => (no_of_features+1,)
    
    def predict(self, x_i):    
#         print(x_i, self.weights)
        weighted_sum = np.dot(x_i, self.weights)  #weighted sum of weights and input feature vector
        return 1 if weighted_sum > 0 else 0 #unit step activation
    
    def train(self, X, Y):
        m = len(X)
        X = np.array(X) # dimensions => m*n no_of_training_sample*no_of_features
        ones = np.ones((m,1)) 
        X_with_ones = np.hstack((ones, X)) #adding 1 to every training sample for counting on bias
        for epoch in range(self.epochs):
            print(f"EPOCH {epoch+1}")
            for i in range(m):
                x_i = X_with_ones[i] # transformed input vector 
                y_i = Y[i] # given label
                y_hat = self.predict(x_i) # predicted label
                self.weights += (self.learning_rate)*(y_i-y_hat)*(x_i) #updating the weights based on the predictions of the model
                print(x_i, y_i, y_hat)
                
            print("")
        
        
class GeneticAlgorithm:
    def __init__(self, no_of_matings, no_of_generations, mutation_threshold):
        self.no_of_matings = no_of_matings
        self.no_of_generations = no_of_generations
        self.mutation_threshold = mutation_threshold
    def ranking_models(self, individuals, train_X, train_Y):
        def calculate_no_of_correct_predictions(model): #metric used for ranking 
            no_of_correct = 0
            m = len(train_X)
            X = train_X
            X = np.array(X)
            ones = np.ones((m,1)) 
            X_with_ones = np.hstack((ones, X)) # adding ones so that they combat with the bias variable 
            for i in range(m):
                x_i = X_with_ones[i]
                y_i = train_Y[i]
                y_hat = model.predict(x_i)
                if y_hat == train_Y[i]:
                    no_of_correct += 1
    #         print(no_of_correct)
            return 1000-(m-no_of_correct)
            #using inbuilt python function for ranking the models according to the no of sample they are able to fit 
        return sorted([(i, calculate_no_of_correct_predictions(i)) for i in individuals],key=lambda x: x[1], reverse=True)
    
    def mutate_the_weights_of_the_current_model(self, W, threshold, flag):
        '''mutate the value of weights w.r.t a threshold. If threshold is greater
        than the current output value then update the weight'''
        def change_weight(c):
            if flag:
                return 0.001*random.randint(-10,10)
            else:return 0.001*random.randint(-10,10)

        n = len(W)
        for i in range(n):
            W[i] = change_weight(W[i]) if random.uniform(0,1)<=threshold else W[i]   

        return W

    def produce_child_with_parents(self, W1, W2):
        '''generate a index and split the parent models weights according to it'''
        assert len(W1) == len(W2)
        n = len(W1)
        split_index = random.randint(0+1, n-1)
        new_W1 = np.concatenate((W1[:split_index], W2[split_index:]))
        new_W2 = np.concatenate((W1[split_index:], W2[:split_index]))
        return new_W1, new_W2
    
    def find_the_fittest(self, initial_models, individual_indexes, training_inputs, labels, flag):    

        for generation in range(0, self.no_of_generations):
            new_models = initial_models.copy()
            for _ in range(0, self.no_of_matings): #mating the parent models
                models_to_mate = np.random.choice( individual_indexes, size=2, replace=False)
                #choose any 2 random models
                #producing the offsprings
                offspring1 = Perceptron(4, 0.001, 10)
                offspring2 = Perceptron(4, 0.001, 10)
                offspring1.weights, offspring2.weights = self.produce_child_with_parents(initial_models[models_to_mate[0]].weights,initial_models[models_to_mate[1]].weights)

                new_models.append(offspring1)
                new_models.append(offspring2)

            #mutating the model weights
            for model in new_models:
                model.weights = self.mutate_the_weights_of_the_current_model(model.weights, self.mutation_threshold, flag)

            #ranking them
            ranked_models = self.ranking_models(new_models, training_inputs, labels)

            print("Best model weights",ranked_models[0][0].weights," Score = ", ranked_models[0][1])
            if ranked_models[0][1] == 1000:
                break
            initial_models = [ranked_models[0][0],ranked_models[1][0] ]

            models_that_survived = np.random.choice(range(2, len(ranked_models)), size=2, replace=False)

            for i in models_that_survived:
                initial_models.append(ranked_models[i][0])



if __name__ == "__main__":
    initial_models_no = 4
    individual_indexes = list(range(0, initial_models_no))
    initial_models = [ Perceptron(4, 0.001, 10) for i in range(0, initial_models_no) ]
    training_inputs = [[0, 0, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0],
            [0, 0, 1, 1],[0, 1, 0, 0],[0, 1, 0, 1],
            [0, 1, 1, 0],[0, 1, 1, 1],[1, 0, 0, 0],
            [1, 0, 0, 1],[1, 0, 1, 0],[1, 0, 1, 1],
            [1, 1, 0, 0],[1, 1, 0, 1],[1, 1, 1, 0],
            [1, 1, 1, 1],]
    labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    genetic_algo = GeneticAlgorithm(5, 200, 0.5)
    genetic_algo.find_the_fittest(initial_models, individual_indexes, training_inputs, labels, True)