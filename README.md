## Documentation 

### Neural Network 
writing code for forward and backward propagation of neural network using NumPy
The model has been tested on digit recognition data. By training on 10,000 images the prediction accuracy is around 89%.
To achieve this accuracy the batch size should be small (around 10 to 20). 

| Activation function | Sigmoid |
| ------------------ | -------------- |
| Loss function | Least mean squares(LMS) |

The `predict` function uses sigmoid activations to predict the labels for the given data.

### Genetic Algorithm (GA)
Neural network weights can be trained using genetic algorithms. In this case, a perceptron has been trained. Steps in the corresponding GA are
1. Create the initial perceptrons using random weights
2. Produce the offspring models from the parent models and add them to this current generation. (cross over some of the weights)
3. Mutate the weights
4. Rank the models of this generation using fitness function (no of correct labels is the metric)
5. Keep some of the best models and discard the rest.
6. Check if the expected performance has been achieved otherwise repeat the steps.

The perceptron has been able to fit AND and OR training data using this algorithm. Sometimes it cannot fit the training data because of the initial random weights or the mutation threshold. 

### Task Scheduler 
When tasks arrive at the cloud provider, the manager server scheduler the tasks based on the resources available.
Here are some of the algorithms which the manager server might use to schedule the tasks
The following algorithms are usedd for online scheduling
1. Oppurtunistic load balancing (OLB)
2. Minimum completion time (MCT)
3. Minimum execution time (MET)
4. K percent best (KPB)
5. Switching (SA)

And for offline scheduling we have 
1. Min-Min
2. Min-Max
3. Sufferage 

By using algorithms on some standard datasets the following results have been obtained, 512 tasks and 16 cloud instances.
Results for online algorithms.
| Dataset | OLB | MET | MET | KPB (k=20%) | SA (low=0.6, high=0.9) |
| --------------- | --------------- | ----------------- | ------------ | -------------- | ------------- |
| u_c_hihi |  14376662.176 |   11422624.494 |   47472299.43 |   22971511.17 |   47472299.43 |  
| u_c_hilo |  221051.824 |   185887.404 |   1185092.969 |   481097.779 |   1185092.969 |  
| u_c_lohi |  477357.02 |   378303.625 |   1453098.004 |   714831.099 |   1453098.004 |  
| u_c_lolo |  7306.596 |   6360.055 |   39582.297 |   16120.163 |   39582.297 |  
| u_i_hihi |  26102017.618 |   4413582.982 |   4508506.792 |   4209797.013 |   4508506.792 |  
| u_i_hilo |  272785.201 |   94855.913 |   96610.481 |   89697.971 |   96610.481 |  
| u_i_lohi |  833605.655 |   143816.094 |   185694.594 |   136820.796 |   185694.594 |  
| u_i_lolo |  8938.027 |   3137.35 |   3399.285 |   3011.106 |   3399.285 |  
| u_s_hihi |  19464875.91 |   6693923.896 |   25162058.136 |   6942283.071 |   25162058.136 |  
| u_s_hilo |  250362.114 |   126587.591 |   605363.773 |   140343.822 |   605363.773 |  
| u_s_lohi |  603231.467 |   186151.286 |   674689.536 |   199945.752 |   674689.536 |  
| u_s_lolo |  8938.389 |   4436.118 |   21042.413 |   5007.078 |   21042.413 |

Results for offline algorithms .
| Dataset | Min-Min | Min-Max | Sufferage |
| --------------- | -------------- | ------------ | ----------- |
| u_c_hihi |  8460675.003 |   12385671.828 |   10249172.885 |  
| u_c_hilo |  161805.434 |   204054.589 |   168982.602 |  
| u_c_lohi |  275837.356 |   392566.686 |   337121.461 |  
| u_c_lolo |  5441.428 |   6945.362 |   5658.538 |  
| u_i_hihi |  3513919.281 |   8018378.072 |   3306818.943 |  
| u_i_hilo |  80755.679 |   151923.835 |   77589.104 |  
| u_i_lohi |  120517.709 |   251528.848 |   114578.915 |  
| u_i_lolo |  2785.645 |   5177.709 |   2639.318 |  
| u_s_hihi |  5160342.819 |   9208811.495 |   5121953.621 |  
| u_s_hilo |  104375.164 |   172822.698 |   102499.893 |  
| u_s_lohi |  140284.489 |   282085.731 |   150297.125 |  
| u_s_lolo |  3806.828 |   6232.242 |   3846.469 |    
