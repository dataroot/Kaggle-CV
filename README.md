# Recommender Engine for CareerVillage
## Description of solution
Description of solution:

- Method for taking into account all the possible content features using custom Neural Network Architecture. Allows automatically find a probability mapping between question and professional entities in all of the combinations (que-que, que-pro, pro-que, pro-pro). It is trained in a way that even without any information about professional and his activity, it still recommends questions, and breaks cold start problem having its own internal scoring.
- Method activity_filter for filtering out all the inactive professionals in order to send immediate emails to professionals who are most likely answer the question, and answer it fast
- Methods spam_filter and email_filter for sending emails which allows to handle spam problem and controls frequency and amount of emails for each professional based on his personal emails reaction type and speed (here reaction==answer)
- The method which tries to activate inactive and "cold" professionals by breaking activity_filter and email_filter with some probability. This is needed for making an attempt to activate the user or remind him about relevant questions which were already sent to him.


# Table Of Contents
-  [Intro](#intro)
-  [Try it first](#try-it-first)
-  [Structure](#structure)
-  [Add new feature](#add-new-feature)
-  [Future Plans](#future-plans)


# Intro
This is a code which represenst our [solution](https://www.kaggle.com/ididur/nn-based-recommender-engine) in more viable for usage way.

# Try it first
In order to get more familiar with the code and how it works let's run it.

You are free to create virtualenv here. We are assuming you're using 3.6 version of python.

Clone repository
```bash
git clone https://github.com/dataroot/Kaggle-CV.git
cd Kaggle-CV
```

##### Important note
All further code snippents with or without `cd`'s are made under assumption you're in project root folder `Kaggle-CV`

Install requirements
```bash
pip install -r requirements.txt
```

In order to run it you will also need to download nltk stopwords. So in python interpreted run:
```python
import nltk
nltk.download('stopwords')
```

Put the data in place

Download the data from https://www.kaggle.com/c/data-science-for-good-careervillage/data.
Create `data` folder and move all the csv files there.

Now we are ready to run our flask app and check how predictor works:
```bash
python app.py
```

Wait a minute while application initializes.
Go to the http://0.0.0.0:8000, and check how it works, this demo is also availbale here: https://careervillage-kaggle.datarootlabs.com.

# Structure

Folder structure
--------------

```
├──  data               - this folder contains csv files from competition
│
│
├── demo_data           - this folder contains sample professionals and their tags, used in professionals selector for demo
│
│
├── dump                - this folder contains preprocessed data and model weights
│
│
├── models              - this folder contains all the models, so you can easily add new architectures and try them
│   └── concat.py
│   └── distance.py
│   └── encoder.py
│   └── simple.py
│   
│   
├── preprocessors        - this folder contains all the feature preprocessors, you can create new one inherited from BaseProc class
│   └── baseproc.py
│   └── proproc.py
│   └── queproc.py
│   └── stuproc.py
│   
│   
├── recommender            - recommendation engine folder
│    └── activity.py  	   - here are all activity filters described in details in our kernel notebook
│    └── demo.py  	       - python file which shows how Predictor works, run with `python demo.py`
│    └── predictor.py  	   - contains two classes Predictor for content based recommendations, and Formatter for nice outputs
│    └── eg_que_to_pro.py  - epsilon-greedy questions to proffesional recommender
│ 
│ 
├── train                  - directory containing Batch generator and training script
│    └── generator.py      - BatchGenerator for generating training data for models
│    └── main.py           - train scipt, run with `python main.py`
│ 
│ 
├── utils                  - useful utils
│    └── importance.py     - 
│    └── utils.py
│ 
│ 
├── views                  - views for demo used by flask app
│    └── index.html
│ 
│ 
└──  app.py                - flask app with a few routes
```

# Add new feature

As you can see, it is easy to add new architectures, npl algorithms, processors, utils and test it extending flask app. 

But it is really important to explain on integration of new feature:
So, if you wish to add a new feature, there are two possible options:

- New feature is time-independent. In that case, first you can either:
	- Add new feature as a column of target entity's dataframe you passing to transform()
	- Add a calculation of new feature on the top of transform() method of target entity's processor class
		- For example, in case of target entity being student, we have added student's state as a feature with the following line, which is the first in StuPruc's transform() method: `stu['students_state'] = stu['students_location'].apply(lambda s: str(s).split(', ')[-1])` 
		Then, you have to add a new feature to `self.features` dict in target entity's class constructor and to specify its type:
	- If the feature's type is categorical, you have to append tuple of feature's name and number of its most popular categories to consider
	- If the feature is numerical, append its name to either 'mean' or 'zero,' depending on value to fill its NaNs
- New feature is time-dependent. Then, to properly train model, you need to:
	- Update `self.features` just like in the previous case
	- Specify its value default value below `#DEFAULT CASE`, which is used in case professional or student do not have answers or questions respectively at a time
		- For example, default value of student's number of asked questions is zero and it is specified in line: `new = {'students_questions_asked': 0, ...`
	- Specify formulas to update it on current timestamp below # UPDATE RULES
		- For example, the value of the number of student's asked questions is updated in line `new['students_questions_asked'] += 1`
	- If a feature needs to be averaged, add its name to list below # NORMALIZE AVERAGE FEATURES


# Future Plans

- Model environment
- Train VAE
- Engineer reward function in order to solve:
	- No question left behind
	- Don’t churn the professionals (too much email → feels like spam → people leave) (partially solved with our currenlt approach, but there are still custom parameters which can be learned and adjusted automatically)
	- Speed to first answer
- Train the agent
