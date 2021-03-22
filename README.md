# OkCupid-Clustering-NLP-NER

### This project was completed by the following team:

* Peter Mankiewich
* Chenzhi Pan
* Yi-Shuan Wang

### Project Summary
OkCupid is a dating application that allows users to make a profile by answering a series of questions (your gender, ethnicity, smoking habits, etc.). Then, the user can swipe on other profiles, and view the percent match, which is a metric automatically calculated based on the contents of both profiles.

Amid a global pandemic, it can be difficult if not impossible to engage in traditional dating while still staying safe, and adhering to public health guidelines. Dating applications, such as the popular OkCupid, become a go-to solution, allowing users to meet remotely, and even plan a date over Zoom without coming into contact with another individual. However, understanding the virtual dating landscape ahead of time can help get the best results from the application. The dataset, which includes features about a user’s profile, can help us understand the types of users that are on the platform. By creating distinct clusters of users, we can help someone make better decisions while swiping, leading to better results for match seekers. Gaining these additional insights is particularly useful given the fact that users only have a limited number of swipes they can make per day, and of course, our time is valuable.


### Text Analysis

Our dataset contains 10 different text columns each containing user generated answers to various questions presented in the app when users create their OkCupid profiles. In addition to using the categorical and numeric variables available to us, we would also like to leverage this text data in our models. 

The first step involved cleaning these columns to ensure that all of the text data could be used in the clustering. Using the python package langdetect, we iterated through all 10 essay columns for all of our users, and determined if the text was in English. While we found that all of the text was English, the process did uncover some strange values, including links and special characters that we removed in order to not affect the later text analysis steps. As mentioned previously, each column corresponds to a different question, however, there were discrepancies in which columns represented which question, and so we decided to concatenate the text for each question together and look at our analysis on an aggregated basis for each user. 

We then decided to calculate the sentiment and subjectivity for each piece of text. These features could be meaningful in the later clustering steps, allowing us to classify users by how positive they are, and how opinionated their answers are to the questions in the app. 



![entities labeled](https://uploads-ssl.webflow.com/5fee20a283f67d529b674a99/6058d7ecd04c130e329a2221_entities_expanded.png)



While we were interested in the general characteristics of the text, we also wanted to understand what sort of topics users were talking about in their responses. In order to do this, we used Spacy to find different entities, or specific subjects in the text. For example, if a user travels a lot, then Spacy would automatically recognize words like “Paris”, or “Japan”, as locations (GPE). This data can help us to understand users who talk about similar topics and have similar interests. Above is a screenshot of one of the responses written by a user. 

There are a number of useful entities identified, including a number of locations that might give us the idea that this person enjoys traveling. It is also important to note that some of the entities are not correctly labeled. For example, the number 1 is labeled as money when given the context, it is not money. Also, Spacy was unable to recognize New York as a location.

 While we feel that these entities are useful in understanding a user’s interests, and their level of seriousness when it comes to writing meaningful responses, we felt like the model would benefit from more domain-specific entities. We decided that we would identify hobbies and interests in the responses not only to match users with similar interests together, but to understand if a user is serious about using the app, and sharing detailed information about themselves. 

We pulled a dataset off of Kaggle with a list of 666 different hobbies, and then checked this list against each of the written responses. For the example given above, we found the below hobbies/interests. In addition to searching the string for the given hobbies, we also used Spacy lemmatization; if the hobby was “acting”, then we would also search the documents for “act”. 

['acting', 'sports', 'travel', 'traveling', 'traveling', 'acting', 'arts', 'drama', 'games', 'gaming', 'internet', 'shooting', 'sports', 'traveling']

It is important to note that while this method does give some useful results, it does have its limitations, and could be improved upon in the future. For example, the word “act” was compared to the response, and since the beginning of some of the words in the document start with “act”, this user was marked as talking about acting even though those words have nothing to do with acting. On the other hand, the word “travel” was picked up, which is really important for this user since they talked a lot about this in their responses.

When it comes to creating the clusters, we generated a column which is the count of the number of hobbies for each user. We felt that this would be useful because it indicates how detailed the user was in their response, and if they have a lot of interests.

Below, we can see a wordcloud of all of the hobbies that existed across the dataset. Based on the wordcloud, we can see that traveling is a really popular interest across the users in our dataset.

![wordcloud](https://uploads-ssl.webflow.com/5fee20a283f67d529b674a99/6058d7eb54a9cc3b6dfe6e1a_wordcloud.png)

### Dimension Reduction and K-Means Clustering
After performing text analysis and NER, we were left with a variety of additional features that we would like to use to generate clusters of our users. First, we performed a dimensionality reduction process using TSNE (t-Distributed Stochastic Neighbor Embedding). This process converted our original dataframe into just two features. Dimensionality reduction can help us to isolate the signal in the data, retaining the overall variance while reducing the noise. We then performed KMeans clustering on the two TSNE columns. The silhouette score and inertia values allowed us to choose an optimal number of clusters that were distinct from one another. In the next section, we explore these clusters in detail.

### Cluster Summaries
We generated a number of different clustering models for various splits of the data. We first split the data by gender and created clusters for men and then for women. We also split the data by people who are above and below the age of 30, and then created clusters for these two subsets. In order to determine the optimal number of clusters, we examined both the silhouette and inertia scores, and combined this knowledge with what we felt would be the most useful number of clusters given the context of the problem.

When examining the seven clusters that we created for just women, we observed some interesting trends and distinct groups that would be useful for OkCupid. We can see that overall, the different diets are distinct between the groups. For example, group one is distinctly people who eat Kosher. In addition, we have a group of older people with children (this was true for a few other subsets as well). We can observe that people who put more effort into their essay (those with lots of entities and hobbies mentioned) on average tend to have a higher income and a good job. Also, relationship status and orientation were very distinct across groups. 

![first cluster](https://uploads-ssl.webflow.com/5fee20a283f67d529b674a99/6058d7d49a4b270e31e41e5d_18.png)

When it comes to the male group, we see some distinct differences in how the seven groups are arranged. While one cluster tends to have older people with kids, we also see that filling out the essays completely is not as strongly associated with high income and a good job (compared to the analysis of females). Again, being married does not seem to be strongly grouped with putting a lot of effort into the essays. Unlike for females, the diets are less distinct, and Halal and Kosher seem to be more grouped together in the same cluster. 

![second cluster](https://uploads-ssl.webflow.com/5fee20a283f67d529b674a99/6058d7d3d04c13949f9a21fd_19.png)

Next, we created clusters for only the subset of people who are above 30, and the subset that are below 30. For those younger than 30 years, we can again see that there is a cluster who are married who don’t put a lot of effort into their essays. In this group, people who do drugs and drink are also in a group who spend a lot of time on their essays and are potentially more serious. Also, we see that people with a higher education level are more serious about their profiles (which is an observation similar to some made in previous clusters). There is also a cluster of people who are gay, who put an average amount of effort into their essays.

![third cluster](https://uploads-ssl.webflow.com/5fee20a283f67d529b674a99/6058d7d16b10b5209ef96613_20.png)

Finally, we performed cluster analysis on people who are over the age of 30. In this case, people with a high income are less likely to be grouped with people who put effort into their essays. Again, there is a group of slightly older people with children. There is also a group that is gay and has a positive sentiment.

![fourth cluster](https://uploads-ssl.webflow.com/5fee20a283f67d529b674a99/6058d7d23cbb95384a0d0de0_21.png)

We developed some descriptive phrases to describe different groups in the analysis. These groups that appear in many of the cluster models could be used by OkCupid to provide people with good match recommendations, and improve the overall user experience. 

* self-made entrepreneurs 
* healthy lifestyle
* single parent
* dog/cat people
* drinking-smoking-drugging
* gay and positive outlook
* married-and-looking-for-an-affair
* in-a-relationship-but-always-looking-for-the-next-romance