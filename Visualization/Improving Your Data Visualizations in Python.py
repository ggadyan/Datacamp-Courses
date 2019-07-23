# =============================================================================
# Chapter 1
# =============================================================================


# Video 1
# Highlighting data
# importing data
import os 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
os.chdir("/Users/Asus UX310/Desktop/DataCamp courses/data")
pollution= pd.read_csv("pollution_wide.csv")
pollution.head()
pollution.city.unique()
# highlighting data
cinci_pollution = pollution[pollution.city =='Cincinnati']
# Make an array of colors based upon if a row is a given day
cinci_colors = ['orangered' if day == 38 else 'steelblue'
for day in cinci_pollution.day]
# Plot with additional scatter plot argument facecolors
p = sns.regplot(x='NO2', y='SO2', data = cinci_pollution, fit_reg=False,
scatter_kws={'facecolors': cinci_colors, 'alpha': 0.7})
#programmatically creating a highlight
houston_pollution = pollution[pollution.city  ==  'Houston'].copy()
# Find the highest observed O3 value
max_O3 = houston_pollution.O3.max()
# Make a column that denotes which day had highest O3
houston_pollution['point type'] = ['Highest O3 Day' if O3  ==  max_O3 else 'Others' for O3 in houston_pollution.O3]
# Encode the hue of the points with the O3 generated column
sns.scatterplot(x = 'NO2', y = 'SO2', hue = 'point type',data = houston_pollution)
plt.show()


# Video 2
# Comparing groups
pollution_nov = pollution[pollution.month == 10]
sns.distplot(pollution_nov[pollution_nov.city =='Denver'].O3, hist=False, color ='red')
sns.distplot(pollution_nov[pollution_nov.city !='Denver'].O3, hist=False)
# Enable rugplot
sns.distplot(pollution_nov[pollution_nov.city =='Denver'].O3,hist=False, color='red', rug=True )
sns.distplot(pollution_nov[pollution_nov.city !='Denver'].O3, hist=False)
# rug plot
sns.distplot(pollution[pollution.city == 'Vandenberg Air Force Base'].O3, label = 'Vandenberg', hist = False, color = 'steelblue', rug = True)
sns.distplot(pollution[pollution.city != 'Vandenberg Air Force Base'].O3, label = 'Other cities', hist = False, color = 'gray')
plt.show()
# kde plot with shade
# Filter dataset to the year 2012
sns.kdeplot(pollution[pollution.year == 2012].O3, shade = True,label = '2012')
# Filter dataset to everything except the year 2012
sns.kdeplot(pollution[pollution.year != 2012].O3, shade = True, label = 'other years')
plt.show()
# the beeswarm plot #1
pollution_nov = pollution[pollution.month == 10]
sns.swarmplot(y="city", x="O3", data=pollution_nov, size=4)
plt.xlabel("Ozone (O3)")
# the beeswarm plot #2
# Filter data to just March
pollution_mar = pollution[pollution.month == 3]
# Plot beeswarm with x as O3
sns.swarmplot(y = "city", x = 'O3',  data = pollution_mar,size = 3)
plt.title('March Ozone levels by city')
plt.show()


# Video 3
# Annotations
# version 1
sns.scatterplot(x='NO2', y='SO2', data = houston_pollution)
# X and Y location of outlier and text
plt.text(13,33,'Look at this outlier', fontdict = {'ha': 'left','size': 'x-large'})
# version 2
sns.scatterplot(x='NO2', y='SO2', data = houston_pollution)
# Arrow start and annotation location
plt.annotate('A buried point to look at', xy=(45.5,11.8), xytext=(60,22), arrowprops={'facecolor':'grey','width': 3}, backgroundcolor ='white' )
# version 3
# Query and filter to New Years in Long Beach
jan_pollution = pollution.query("(month  ==  1) & (year  ==  2012)")
lb_newyears = jan_pollution.query("(day  ==  1) & (city  ==  'Long Beach')")
sns.scatterplot(x = 'CO', y = 'NO2', data = jan_pollution)
plt.annotate('Long Beach New Years', xy = (lb_newyears.CO, lb_newyears.NO2), xytext = (2, 15),  arrowprops = {'facecolor':'gray', 'width': 3, 'shrink': 0.03}, backgroundcolor = 'white')
plt.show()


# =============================================================================
# Chapter 2
# =============================================================================


# Video 1
# Color in visualizations

#sns.barplot(x = values, y = ids, color ='cadetblue', edgecolor ='black')
# Hard to read scatter of CO and NO2 w/ color mapped to city
# sns.scatterplot('CO', 'NO2',alpha = 0.2, hue = 'city', data = pollution)
# Setup a facet grid to separate the cities apart
g = sns.FacetGrid(data = pollution, col = 'city', col_wrap = 3)
# Map sns.scatterplot to create separate city scatter plots
g.map(sns.scatterplot, 'CO', 'NO2', alpha = 0.2)
plt.show()


# Video 2
# Continuous color palettes
# Filter the data
cinci_2014 = pollution.query("city  ==  'Cincinnati' & year  ==  2014")
# Define a custom continuous color palette
color_palette = sns.light_palette('orangered',as_cmap = True)
# Plot mapping the color of the points with custom palette
sns.scatterplot(x = 'CO', y = 'NO2', hue = 'O3',  data = cinci_2014, palette = color_palette)
plt.show()


# Video 3
# Categorical palettes
# Assign a new column to dataframe the desired combos
pollution['interesting cities'] = [x if x in ['Long Beach','Cincinnati'] else 'other' for x in pollution['city'] ]
sns.scatterplot(x="NO2", y="SO2", hue ='interesting cities', palette='Set2',data=pollution.query('year == 2014 & month == 12'))

# ordinal palettes
colorbrewer_palettes = ['Reds','Blues','YlOrBr','PuBuGn','GnBu','Greys']
for i, pal in enumerate(colorbrewer_palettes):
    sns.palplot(pal=sns.color_palette(pal, n_colors=i+4))
# ordinal plot
# Make a tertials column using qcut()
pollution['NO2 Tertial'] = pd.qcut(pollution['NO2'], 3, labels = False)
# Plot colored by the computer tertials
sns.scatterplot(x="CO", y="SO2", hue='NO2 Tertial', palette="OrRd", data=pollution.query("city =='Long Beach' & year == 2014"))

# line graph for pollution per city
# Filter our data to Jan 2013
pollution_jan13 = pollution.query('year  ==  2013 & month  ==  1')
# Color lines by the city and use custom ColorBrewer palette
sns.lineplot(x = "day",  y = "CO", hue = "city", palette = "Set2",  linewidth = 3, data = pollution_jan13)
plt.show()



# =============================================================================
# Chapter 3
# =============================================================================
# Video 1
# Point estimate intervals

# Video 2
# Confidence bands
# plt.fill_between(x='day', y1='lower', y2='upper', data=cinci_so2)
# plt.plot('day','mean','w-', alpha=0.5, data=data)
# alpha helps see what is underneath the overlaps (using color helps too)

# Video 3
# Beyond 95%

# Video 3
# Visualizing the bootstrap
# =============================================================================
# Chapter 4
# =============================================================================
# Video 1
# First explorations
# pd.plotting.scatter_matrix(pollution, alpha = 0.2);

# Video 2
# Exploring the patterns

# Video 3
# Making your visualizations efficient

# Video 4
# Tweaking your plots

# Video 5
# Wrap-Up






























































































