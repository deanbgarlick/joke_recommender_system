# joke_recommender_system

Repo to implement the building of a joke recommendation system in PySpark on data belonging to the Jester Dataset avaliable at url http://eigentaste.berkeley.edu/dataset/ 

To build the recommendation system the following PySpark ML classes will be fitted:
 (1) Alternating Least Squares (ALS)
 (2) Latent Dirichlet Allocation (LDA)
 
From (1) we can obtain a predicted ranking of each joke's score for each user. We will use (2) to obtain a topical clustering of the jokes. The initial recommendation engine will then provide a number of most recommended jokes from topics for any given user.

The dataset citation is: "Eigentaste: A Constant Time Collaborative Filtering Algorithm. Ken Goldberg, Theresa Roeder, Dhruv Gupta, and Chris Perkins. Information Retrieval, 4(2), 133-151. July 2001."
