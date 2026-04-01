CGAN Travel Recommendation System

This system is a travel recommendation system made with a Conditional Generative Adversarial Network (CGAN).
The main contribution of this system is to solve the cold-start challenge of sparse data travelers (i.e., travelers with no travel history, only traveling for a short time with places).

This system has a novel algorithm, an 'Anchor-and-Expand' algorithm for cold-start users who have 'zero' travel history. Yes, zero (none). You can check out the dataset and scroll all the way down and find userID: 111111. That is the cold-start user with no travel history.
This algorithm works by: 
1. Gather all information from a traveler with history, including visit frequency, GPS coordinates, theme/subtheme of POI places, users' preference of the POI theme/subtheme, and sequence of patterns from all users.
2. Train with CGAN. CGAN analyzes the information from the user and makes a concept of what kind of preference this user likes, and looks through the sequence of POI with route pattern and simultaneously finds out where the user hasn't visited + where the other similar users went.
3. Then it searches for an anchor POI as a start point. And the candidate of the start point is all POIs inside the datasets.
4. After it searches for the anchor POI, it starts to expand the route with the specific pattern where that user has been before and recommends it to the new user. Duplicate POIs and routes are not generated.
5. Finally, it shows the routes with POIs and shows them on a map. You can go ahead and check out the generated route for the cold-start user from the cold-start generated routes folder. I've tested out with 5 cities (Edinburgh, Glasgow, Melbourne, Toronto, Osaka) and it works like a charm.

You can find the results by clicking the 5 cities folder (Edinburgh, Melbourne, Glasgow, Toronto, Osaka).
This includes the training results of CGAN, t-SNE, and dimensions.

This system was executed on Python 3.11.9 on Windows 11.

To execute the system, click the CGAN_RecSys.py and install the necessary libraries, and run it.
