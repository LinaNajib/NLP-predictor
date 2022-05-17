#This assignment will make predictions using natural language processing techniques using Yelp dataset.

# ### 1. Import data (yelp.csv).


import pandas as pd


yelp_data=pd.read_csv('yelp.csv',sep=',')


yelp_data.head(3)


yelp_data.describe()


yelp_data.info()


yelp_data.shape


# ### 2. Visualize the voting columns by either histogram, seaborn, or countplot.


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')



plt.figure(figsize=(4,4))
sns.countplot(x="stars", data=yelp_data)
plt.title('Voting Histogram')
plt.xlabel('Stars')
plt.ylabel('Frequency')


# ### We will add a new column "text_len", and add the length of each review into it.


yelp_data['text_len']= yelp_data['text'].apply(len)  


# ### Histogram will show  the frequency of 'text_len' column.


yelp_data['text_len'].plot.hist(bins=50)


# ### We can notice from using FacetGrid that there are similarities between the histograms. All of them have the same range, are skewed to the left side, and all their peaks are near 500 of text_length. But for stars 4- and 5- histograms have much higher peaks than the rest.


g = sns.FacetGrid(data=yelp_data, col='stars')
g.map(plt.hist, 'text_len', bins=50)


# ### We will work on the column 'useful' as an example,  and we can try with the other two columns: 'cool' and 'funny' as well.


ax = sns.countplot(x="useful", hue="stars", data=yelp_data)
plt.title('Countplot of Voting')
plt.xlabel('useful')
plt.legend(loc='upper right',title='stars')


x1=yelp_data['stars']
x2=yelp_data['useful']
plt.plot(x1, x2,".")


# ### From the boxplot below, we can notice that it has more information because it shows us the outliers and the actual range of the data in the column "useful". (we can try with the other two columns: 'cool' and 'funny').


sns.boxplot(x='stars', y='useful', data=yelp_data)


# ### 3. Make the dataframe (feature = text and label = stars) for the reviews with stars greater than zero.



#df = pd.DataFrame(data=yelp_data, columns=['text','stars'])
new_yelp = pd.DataFrame(data=yelp_data, columns=['text','stars','text_len'])


# ### Filter the reviews by stars greater than zero:


new_yelp= new_yelp[new_yelp.stars > 0]



new_yelp.shape



new_yelp.head(5)


# ### 4. Tokenize the text and generate the word vector.


import re
import nltk
#import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize

#nltk.download('punkt')       #---punctuation library-------DONE
#nltk.download('stopwords')   #---stopwords library------DONE


# ### Convert the text into lower case, split it, remove the stopwords from it, then plot the frequency distribution:


word_list=[]
i=0
for txt in new_yelp['text']:
     w_list = re.sub(r'[^a-z]',' ',txt.lower()).split()   #re:regular expression
     new_yelp.loc[i,'text']=' '.join(w_list)
     w_list=[w for w in w_list if w not in stopwords.words('english')]
     word_list+=w_list
     i+=1



all_words = nltk.FreqDist(word_list)
all_words.plot(20,cumulative=False)



len(word_list)



all_words.most_common(20)



new_yelp.head(n=5)


# ### Select the most 3000 common words and consider them our vocabulary:

# In[15]:


#repeated line!!!!!!!!!!!!!
#all_words=nltk.FreqDist(word_list)



word_common=all_words.most_common(3000) 



#no need for this code!!!!!!!!!!!!!


#w[0] contains the name of the word only, w[1] contains the number of it
#word_features=[w[0] for w in word_common]  


def find_feature(document):
    features={}
    words=document.split()
    #check if the word is in our vocabulary or not
    for w in word_common:   
        features[w[0]]=(w[0] in words) #this is a logical condition(if w[0] in words then TRUE)
    return features

Feature_set=[]
for i, row in enumerate(new_yelp.values):
    #it is spliting the row contents into its corresponding features/columns
    text,stars,text_len=row  
    #call the function, and save/append the features with thier labels
    Feature_set.append((find_feature(text),stars)) 


len(Feature_set)


# In[20]:


Feature_set[1:2]


# ### 5. Divide the dataframe into train and test sets.


train_set=Feature_set[:7500]  #we are making the first 7500 row as training set=75%
test_set=Feature_set[7500:]  #and the remaining rows are testing set=25%


# ### 6. Fit a classifier on the train data.


classifier=nltk.NaiveBayesClassifier.train(train_set)


# ### Predict the label of the test data.

# ### Generate the confusion matrix or accuracy of the predicted and real label of the test set.


print("Naive bayes algorithm accuracy percent:",(nltk.classify.accuracy(classifier,test_set))*100)


# In[22]:


# Determine the most relevant features, and display them.

classifier.show_most_informative_features(3000)  


# ### We noticed that the previous result of 'Naive bayes' algorithm accuracy percent: 43.92, which is so little. The reason behind that is that we have 5 stars which mean we have five categories. The solution is to divide the dataset into two categories, for example: merge stars 5 and 4 into one group and consider them as high rating and assign 5 to them, and stars 1, 2, and 3 into another group and consider them as low rating and assign 1 to them. The code and the results are as shown below:


#we used this code when stars=3 is considered "bad":

new_yelp.loc[(new_yelp.stars == 5) | (new_yelp.stars == 4), 'new_stars'] = 5  
new_yelp.loc[(new_yelp.stars <= 3) , 'new_stars'] = 1 



#we used this code when stars=3 is considered "good":

#new_yelp.loc[(new_yelp.stars == 5) | (new_yelp.stars == 4)| (new_yelp.stars == 3), 'new_stars'] = 5  
#new_yelp.loc[(new_yelp.stars <= 2) , 'new_stars'] = 1 


new_yelp.head(n=5)



new_yelp.tail(n=7)


# ### Now, Let's do the same steps but with the" new_stars" column, as shown below:



def find_feature(document):
    features={}
    words=document.split()
    #check if the word is in our vocabulary or not
    for w in word_common:   
        features[w[0]]=(w[0] in words) #this is a logical condition(if w[0] in words then TRUE)
    return features

Feature_set=[]
for i, row in enumerate(new_yelp.values):
    #it is spliting the row contents into its corresponding features/columns
    text,stars,text_len,new_stars=row  
    #call the function, and save/append the features with thier labels
    Feature_set.append((find_feature(text),new_stars)) #use the new column


train_set=Feature_set[:7500]  #we are making the first 7500 row as training set=75%
test_set=Feature_set[7500:]  #and the remaining rows are testing set=25%


classifier=nltk.NaiveBayesClassifier.train(train_set)


# ### The accuracy result when stars=3 is considered "bad":


print("Naive bayes algorithm accuracy percent:",(nltk.classify.accuracy(classifier,test_set))*100)


# ### The accuracy result when stars=3 is considered "good":



print("Naive bayes algorithm accuracy percent:",(nltk.classify.accuracy(classifier,test_set))*100)


# ### In the code below, we can count howmany 3 stars review we have in our dataset:


new_yelp["stars"].value_counts()

