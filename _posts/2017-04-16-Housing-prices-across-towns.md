---
layout: post
title: Housing Prices across university/non-university towns
tags: python, housing prices, introduction to data science in python
---

Hello World!

First post ever on Github Pages!

The project covered in this post is based on Week 4 of UMichigan's [Introduction to Data Science in Python](https://www.coursera.org/learn/python-data-analysis/home/welcome) on Coursera. I have its repository on my [Github](https://github.com/aliciasueyee/Introduction-to-Data-Science-with-Python), and the source data files. I hope to extend this project a little further, perhaps creating linear regression models.

So the point of this project is to compare the impact of recession on housing prices in university towns/non-university towns.

**Hypothesis:** University towns housing prices are *less* affected by recession compared to non-university housing prices.

Few notes:
* A quarter is 3 months; Jan-March is Q1, Apr-Jun is Q2, Jul-Sept is Q3, Oct-Dec is Q4.
* A recession starts when GDP declines in two consecutive quarter, and ends with GDP growth in two consecutive quarters.
* Recession bottom is the quarter with the lowest GDP in a recession.
* A university town is a city with a high percentage of college students compared to the total population of said city. 

**I like to break things down into intermediate steps:**
1. Find out when recession period started & ended.
2. Find out when recession was at the bottom.
3. Make list of university towns.
4. Make dataframe of housing prices.
5. Using housing prices dataframe, calculate ratio of housing prices at start/end of recession period.
6. Using ratio of housing prices, run a t-test to test for a significant difference across university/non-university towns. 

Hope you're still with me!

First things first, import your libraries. You'll need pandas, numpy, and an independent t-test from the scipy.stats library.

{% highlight python %}
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
{% endhighlight %}

We're using the chained value to 2009 dollars, 2000 onward. Get your recession start, in string format:

{% highlight python %}
def get_recession_start():
    gdp = pd.read_excel('gdplev.xls', skiprows = 7, usecols= {'Unnamed: 4', 'Unnamed: 6'})
    gdp = gdp.loc[212:]
    gdp = gdp.rename(columns = {'Unnamed: 4': 'Quarter', 'Unnamed: 6': 'GDP'})
    gdp['GDP'] = pd.to_numeric(gdp['GDP'])
    global gdp
    quarters = []
    for i in range(len(gdp) - 2):
        if (gdp.iloc[i][1] > gdp.iloc[i+1][1]) & (gdp.iloc[i+1][1] > gdp.iloc[i+2][1]):
            quarters.append(gdp.iloc[i+1][0])
    return quarters[0]
{% endhighlight %}

And get the recession end:

{% highlight python %}
def get_recession_end():
    #figured out that gdp[gdp['Quarter'] == '2008q2'].index.tolist() was [245], but for some reason it won't compile right
    gdp2 = gdp.loc[245:]
    recession_end = []
    for i in range(len(gdp2)- 2):
        if (gdp2.iloc[i+2][1] > gdp2.iloc[i+1][1])  & (gdp2.iloc[i+1][1] > gdp2.iloc[i][1]):
            recession_end.append(gdp2.iloc[i+2][0])
    return recession_end[0]
{% endhighlight %}

From recession start to recession end, you need to find the bottom:

{% highlight python %}
def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a 
    string value in a format such as 2005q3'''
    recession_period = gdp.loc[245:]
    recession_min = recession_period[recession_period['GDP'] == recession_period['GDP'].min()]
    return recession_min.values[0][0]
{% endhighlight %}

Get a list of university towns:

{% highlight python %}
def get_list_of_university_towns():
    '''Returns a DataFrame of towns and the states they are in from the 
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
    columns=["State", "RegionName"]  )'''
    with open('university_towns.txt') as file:
        data = []
        for line in file:
            data.append(line[:-1])
    state_town = []
    for line in data:
        if line[-6:] == '[edit]':
            state = line[:-6]
        elif '(' in line:
            town = line[:line.index('(')-1]
            state_town.append([state,town])
        else:
            town = line
            state_town.append([state,town])
    state_college_df = pd.DataFrame(state_town,columns = ['State','RegionName'])
    return state_college_df
{% endhighlight %}

And then convert the raw housing data into quarters.

We're looking for averages of the month of Jan-Mar/Apr-June/July-Sept/Oct-Dec, converted into a dataframe, with a multi-index in the shape of ["State", "RegionName"]. You should have 67 columns and 10, 730 rows. 

{% highlight python %}
def convert_housing_data_to_quarters():
    housingdata_df = pd.read_csv('City_Zhvi_AllHomes.csv')
    #convert two-letter-state to full name of state
    housingdata_df['State'] = housingdata_df['State'].map(states)
    #set index to state, regionname
    housingdata_df.set_index(["State","RegionName"], inplace=True)
    #filter columns by year, only want 2000 to 2016
    housingdata_df = housingdata_df.filter(regex='^20', axis=1)
    #group select columns by quarter, calculates average per quarter
    housingdata_df = housingdata_df.groupby(pd.PeriodIndex(housingdata_df.columns, freq='Q'), axis=1).mean()
    global housingdata_df
    return housingdata_df
{% endhighlight %}

Call all the functions from earlier, and drop the NaN values in the dataframe.

{% highlight python %}
recession_start = get_recession_start()
recession_bottom = get_recession_bottom()
university_towns = get_list_of_university_towns()
housingdata_df = convert_housing_data_to_quarters().dropna(){% endhighlight %}

Make a copy of your housingdata_df, then create a ratio of housing prices:

{% highlight python %}
hdf = housingdata_df.copy()
ratio = pd.DataFrame({'ratio': hdf[recession_start].div(hdf[recession_bottom])})
{% endhighlight %}

This was where I struggled. I could not join ratio as a column on hdf; it returned a 

> DateParseError: Unknown datetime string format, unable to parse ratio.

Remember when we converted the housingdata_df into quarters using PeriodIndex? Ratio was not recognized as a datetime. The solution I chose was to change hdf dataframe's columns into strings, then concatenate ratio to the multiple strings... and then convert it back to a dataframe. 

{% highlight python %}
hdf.columns = hdf.columns.to_series().astype(str)
hdf = pd.concat([hdf, ratio], axis=1)

hdf = pd.DataFrame(hdf)
hdf.reset_index(['State','RegionName'], inplace = True)
{% endhighlight %}

Then splice the dataframe into university town and non-university towns, calculate ratio for each, and dropna:

{% highlight python %}
unitown_priceratio = hdf.loc[list(university_towns.index)]['ratio'].dropna()

# minus/exclude university towns from total dataframe to get non-university towns
nonunitown_priceratio_index = set(hdf.index) - set(unitown_priceratio)

#and then calculate the ratio
nonunitown_priceratio = hdf.loc[list(nonunitown_priceratio_index),:]['ratio'].dropna()
{% endhighlight %}

The last bit is to run the t-test. Skip the next two paragraphs if you're familiar with t-tests.

**So what exactly does t-test do?** 
A t-test tests the difference between the means of two independent (or different conditions) groups. For example, let's say we want to compare the mean income of two different groups of peaches, with fertilizer and without. We know there's going to be a difference in the average weight of the two groups, but is the difference in the average weight ENOUGH/sufficiently large to say that these peaches were drawn from different populations?

A common p-value is p < 0.05; that is, the probability of obtaining this sample data is less than 0.05 <b>IF</b> there is no difference between means between the two groups.  

{% highlight python %}
def run_ttest(a, b):
  #run t-test comparing university town values to non-university town values
    tstat, p = tuple(ttest_ind(a, b))
    
    #return tuple where different = True or False, return p-value, and whether university-town is better or not
    different = p < 0.05
    result = tstat < 0
    better = ["university town", "non-university town"]
    
    return (different, p, better[result])

run_ttest(unitown_priceratio, nonunitown_priceratio)
{% endhighlight %}

This returns True, the p-value, and university town. 

Based on the t-test, we can conclude that the alt-hypothesis is not rejected; ie., housing prices in university towns are less impacted by recession compared to housing prices in non-university towns.
