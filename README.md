#Predicting the Popularity of Code

This was a final project of mine for a data mining course in Fall 2012. Basically, I wanted to see if I could tell popular projects from unpopular ones, and if so, what the differences were. I only looked at Python repos on GitHub (quick shoutout to the GitHub team, who kindly provided a custom db dump for me).

My most interesting result was that highly popular repos (350+ stars) can be differentiated from unpopular ones (3-10 stars) quite well *solely* by looking at relative occurence of AST nodes.

There are more details in my paper:

* final writeup: (Google Doc)[https://docs.google.com/document/d/1MBNpGsrt1jIqcLqxFWOW7m_7j1zvouTu3uyqlfLn7EQ/edit]
* final data: (Google Spreadsheet)[https://docs.google.com/spreadsheet/ccc?key=0ArbW86SpnfA8dHhBcGVybEZFZ3pfd3lZb0w0Nm1WVWc]

## Running everything

The code is far from elegant, but it gets the job done. First, grab the database of repos: (Google Drive)[https://docs.google.com/open?id=0B7bW86SpnfA8Wk1GcmF2R1JtN1E]. This is an sqlite db, and should be named 'erepo.db' and placed in the root of the repo.

Before running any code, you'll need some dependencies. Create a new venv, then run `pip install -r requirements.txt` to get them.

Next, you need to pick your sample size. Edit choose_sample.py as you please (you have to download every repo in the sample, so you probably want to keep it small), then run the script. This creates the classes.py file, which you could then edit if you want (maybe to manually add your own repo).

Next is feature calculation, by running featurecalc.py. This can take a while, and when finished creates features.pickle.

Lastly, you can build the classifier and see how it performs by using run_test.py. summarize_feature_data is something else to try; it shows you the min/max/median/mean/std of all features you calculated.

## Notes
If you want to watch feature calculation progress, you can tail calcfeatures.log. I probably won't edit this or update it, but feel free to contact me if you have questions!


- - -

Copyright 2012 [Simon Weber](http://www.simonmweber.com). 
All code licensed under the MIT license.