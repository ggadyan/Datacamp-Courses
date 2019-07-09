# data visualization with seaborn
#=============================================================================
# Chapter 1
# =============================================================================
#introduction to seaborn
#video 1
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("wines.csv")
fig,ax = plt.subplots()
ax.hist(df['alcohol']) #df["alcohol"].plot.hist()
import seaborn as sns
sns.distplot(df['alcohol'])

#ex.1 
#multiple
#answer:matplotlib

#ex.2
# import all modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Read in the DataFrame
df = pd.read_csv(grant_file)

#ex.3
#part 1
# Display pandas histogram
df['Award_Amount'].plot.hist()
plt.show()
# Clear out the pandas histogram
plt.clf()

#part2
# Display a Seaborn distplot
sns.distplot(df['Award_Amount'])
plt.show()
# Clear the distplot
plt.clf()



#using the distribution plot
#video 2
sns.distplot(df["alcohol"], kde=False, bins=10)
sns.distplot(df_wines["alcohol"], hist=False, rug=True, kde_kws={'shade':True})


#ex.1
# Create a distplot
sns.distplot(df['Award_Amount'],
             kde=False,
             bins=20)
# Display the plot
plt.show()


#ex.2
# Create a distplot of the Award Amount
sns.distplot(df['Award_Amount'],
             hist=False,
             rug=True,
             kde_kws={'shade':True})
# Plot the results
plt.show()


#ex.3
#multiple
#answer:There are a large group of award amounts < $400K


#regression plots in seaborn
#video 3
sns.regplot(x="alcohol", y="pH", data=df)
sns.lmplot(x="quality", y="alcohol", data=df, hue="type")

#ex.1

#part 1
sns.regplot(data=df,
         x="insurance_losses",
         y="premiums")
# Display the plot
plt.show()

#part 2
sns.lmplot(data=df,
         x="insurance_losses",
         y="premiums")
# Display the plot
plt.show()


#ex.2
# Create a regression plot using hue
sns.lmplot(data=df,
           x="insurance_losses",
           y="premiums",
           hue="Region")
# Show the results
plt.show()

#ex.3
multiple rows
sns.lmplot(data=df,
           x="insurance_losses",
           y="premiums",
           row="Region")
# Show the plot
plt.show()

# =============================================================================
# Chapter 1
# =============================================================================
#using seaborn styles
#video 1

sns.set()
df["Tuition"].plot.hist()

sns.set_style()
#examples for styling
for style in ['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']:
    sns.set_style(style)
    sns.distplot(df['Tuition'])
    plt.show()

#revoming axes with despine()
    
sns.set_style('white')
sns.distplot(df['Tuition'])
sns.despine(left=True)  
  

#ex.1
# Plot the pandas histogram
df['fmr_2'].plot.hist()
plt.show()
plt.clf()
# Set the default seaborn style
sns.set()
# Plot the pandas histogram again
df['fmr_2'].plot.hist()
plt.show()
plt.clf()


#ex.2

#part 1
# Plot with a dark style 
sns.set_style('dark')
sns.distplot(df['fmr_2'])
plt.show()
# Clear the figure
plt.clf()

#part 2
# Plot with a dark style 
sns.set_style('whitegrid')
sns.distplot(df['fmr_2'])
plt.show()
# Clear the figure
plt.clf()


#ex.3
# Set the style to white
sns.set_style('white')
# Create a regression plot
sns.lmplot(data=df,
           x='pop2010',
           y='fmr_2')
# Remove the spines
sns.despine(right=True, top=True)
# Show the plot and clear the figure
plt.show()
plt.clf()


#colors in seaborn
#video 2
sns.set(color_codes=True)
sns.distplot(df['Tuition'], color='g')

#palettes
for p in sns.palettes.SEABORN_PALETTES:
    sns.set_palette(p)
    sns.distplot(df['Tuition'])

sns.palplot()
sns.color_palettes()

for p in sns.palettes.SEABORN_PALETTES:
    sns.set_palette(p)
    sns.paplot(sns.color_palette())
    plt.show()

#circular colors (not ordered)
sns.paplot(sns.color_palette("Paired", 12))
#sequential 
sns.paplot(sns.color_palette("Blues", 12))
#diverging
sns.paplot(sns.color_palette("BrBG", 12))


#ex.1
# Set style, enable color code, and create a magenta distplot
sns.set(color_codes=True)
sns.distplot(df['fmr_3'], color='m')
# Show the plot
plt.show()


#ex.2
# Loop through differences between bright and colorblind palettes
for p in ['bright', 'colorblind']:
    sns.set_palette(p)
    sns.distplot(df['fmr_3'])
    plt.show()   
    # Clear the plots    
    plt.clf()


#ex.3
#multiple
#answer: circular


#ex.4

#part 1
# Create the coolwarm palette
sns.palplot(sns.color_palette("Purples", 8))
plt.show()
   
#part 2
# Create the coolwarm palette
sns.palplot(sns.color_palette("husl", 10))
plt.show() 
 
#part 3
# Create the coolwarm palette
sns.palplot(sns.color_palette("coolwarm", 6))
plt.show()    
    

#customizing in matplotlib
#video 3
fig, ax=plt.subplots()
sns.distplot(df["Tuition"], ax=ax)
ax.set(xlabel='Tuition 2013-2014')

#further customizations
fig, ax=plt.subplots()
sns.distplot(df["Tuition"], ax=ax)
ax.set(xlabel="Tuition 2013-14",
       ylabel="Distribution", xlim=(0,50000),
       title="2013-14 Tuition and Fees Distribution")
 
#combining plots
fig, (ax0,ax1)=plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(7,4))
sns.distplot(df["Tuition"], ax=ax0)
sns.distplot(df.query('State=="MN"')['Tuition'], ax=ax1)
ax1.set(xlabel='Tuition (MN)', xlim=(0,70000))
ax1.axvline(x=20000, label='My Budget', linestyle="--")
ax1. legend()


#ex.1
# Create a figure and axes
fig, ax = plt.subplots()
# Plot the distribution of data
sns.distplot(df['fmr_3'], ax=ax)
# Create a more descriptive x axis label
ax.set(xlabel="3 Bedroom Fair Market Rent")
# Show the plot
plt.show()


#ex.2
# Create a figure and axes
fig, ax = plt.subplots()
# Plot the distribution of 1 bedroom rents
sns.distplot(df['fmr_1'], ax=ax)
# Modify the properties of the plot
ax.set(xlabel="1 Bedroom Fair Market Rent",
       xlim=(100,1500),
       title="US Rent")
# Display the plot
plt.show()


#ex.3
# Create a figure and axes. Then plot the data
fig, ax = plt.subplots()
sns.distplot(df['fmr_1'], ax=ax)
# Customize the labels and limits
ax.set(xlabel="1 Bedroom Fair Market Rent", xlim=(100,1500), title="US Rent")
# Add vertical lines for the median and mean
ax.axvline(x=df['fmr_1'].median(), color='m', label='Median', linestyle='--', linewidth=2)
ax.axvline(x=df['fmr_1'].mean(), color='b', label='Mean', linestyle='-', linewidth=2)
# Show the legend and plot the data
ax.legend()
plt.show()


#ex.4
# Create a plot with 1 row and 2 columns that share the y axis label
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True)
# Plot the distribution of 1 bedroom apartments on ax0
sns.distplot(df['fmr_1'], ax=ax0)
ax0.set(xlabel="1 Bedroom Fair Market Rent", xlim=(100,1500))
# Plot the distribution of 2 bedroom apartments on ax1
sns.distplot(df['fmr_2'], ax=ax1)
ax1.set(xlabel="2 Bedroom Fair Market Rent", xlim=(100,1500))
# Display the plot
plt.show()


# =============================================================================
# Chapter 3
# =============================================================================
#categorical plots
#video 1

#1. stripplot, swarmplot : plots of each observation
#2. boxplot, violinplot, lvplot: abstract representation
#3. barplot, pointplot, countplot: statistical estimates

#stripplot
sns.striplot(data=df, y="DRG Definition", x="Average Covered Charges", jitter=True)
#swarmplot
sns.swarmplot(data=df, y="DRG Definition", x="Average Covered Charges")

#boxplot
sns.boxplot(data=df, y="DRG Definition", x="Average Covered Charges")
#violinplot
sns.violinplot(data=df, y="DRG Definition", x="Average Covered Charges")
#lvplot:letter value plot for large datasets
sns.lvplot(data=df, y="DRG Definition", x="Average Covered Charges")

#barplot
sns.barplot(data=df, y="DRG Definition", x="Average Covered Charges", hue="Region")
#pointplot
sns.pointplot(data=df, y="DRG Definition", x="Average Covered Charges", hue="Region")
#countplot
sns.countplot(data=df, y="DRG_Code", hue="Region")


#ex.1

#part 1
# Create the stripplot
sns.stripplot(data=df,
         x='Award_Amount',
         y='Model Selected',
         jitter=True)
plt.show()

#part 2
# Create and display a swarmplot with hue set to the Region
sns.swarmplot(data=df,
         x='Award_Amount',
         y='Model Selected',
         hue='Region')
plt.show()


#ex.2
#part 1
# Create a boxplot
sns.boxplot(data=df,
         x='Award_Amount',
         y='Model Selected')
plt.show()
plt.clf()

#part 2
# Create a violinplot with the husl palette
sns.violinplot(data=df,
         x='Award_Amount',
         y='Model Selected',
         palette='husl')
plt.show()
plt.clf()

#part 3
# Create a lvplot with the Paired palette and the Region column as the hue
sns.lvplot(data=df,
         x='Award_Amount',
         y='Model Selected',
         palette='Paired',
         hue='Region')
plt.show()
plt.clf()

#ex.3

#part 1
# Show a countplot with the number of models used with each region a different color
sns.countplot(data=df,
         y="Model Selected",
         hue="Region")
plt.show()
plt.clf()

#part 2
# Create a pointplot and include the capsize in order to show bars on the confidence interval
sns.pointplot(data=df,
         y='Award_Amount',
         x='Model Selected',
         capsize=.1)
plt.show()
plt.clf()

#part 3
# Create a barplot with each Region shown as a different color
sns.barplot(data=df,
         y='Award_Amount',
         x='Model Selected',
         hue="Region")
plt.show()
plt.clf()

#regression plots
#video 2
sns.regplot(data=df, x="temp", y="total_rentals", marker="+")
sns.residplot(data=df, x="temp", y="total_rentals")
sns.regplot(data=df, x="temp", y="total_rentals", order=2) #second order polynomial function
sns.residplot(data=df, x="temp", y="total_rentals", order=2)

#categorical values
sns.regplot(data=df, x="mnth", y="total_rentals", x_jitter=.1, order=2)
#estimators
sns.regplot(data=df, x="mnth", y="total_rentals", x_estimator=np.mean, order=2)
#binning data
sns.regplot(data=df, x="temp", y="total_rentals", x_bins=4)


#ex.1

#part 1
# Display a regression plot for Tuition
sns.regplot(data=df,
         y='Tuition',
         x="SAT_AVG_ALL",
         marker='^',
         color='g')
plt.show()
plt.clf()

#part 2
# Display the residual plot
sns.residplot(data=df,
          y='Tuition',
          x="SAT_AVG_ALL",
          color='g')
plt.show()
plt.clf()


#ex.2

#part 1
# Plot a regression plot of Tuition and the Percentage of Pell Grants
sns.regplot(data=df,
            y='Tuition',
            x="PCTPELL")
plt.show()
plt.clf()

#part 2
# Create another plot that estimates the tuition by PCTPELL
sns.regplot(data=df,
            y='Tuition',
            x="PCTPELL",
            x_bins=5)
plt.show()
plt.clf()

#part 3
# The final plot should include a line using a 2nd order polynomial
sns.regplot(data=df,
            y='Tuition',
            x="PCTPELL",
            x_bins=5,
            order=2)
plt.show()
plt.clf()


#ex.3

#part 1
# Create a scatter plot by disabling the regression line
sns.regplot(data=df,
            y='Tuition',
            x="UG",
            fit_reg=False)
plt.show()
plt.clf()

#part 2
# Create a scatter plot and bin the data into 5 bins
sns.regplot(data=df,
            y='Tuition',
            x="UG",
            x_bins=5)
plt.show()
plt.clf()

#part 3
# Create a regplot and bin the data into 8 bins
sns.regplot(data=df,
         y='Tuition',
         x="UG",
         x_bins=8)
plt.show()
plt.clf()


#matrix plots
#video 3
pd.crosstab (df["mnth"], df["weekday"], values=df["total_rentals"], aggfunc="mean".round(0))

#heatmap
sns.heatmap(d.crosstab (df["mnth"], df["weekday"], values=df["total_rentals"], aggfunc="mean"))
#customize heatmap
sns.heatmap(df_crosstab, annot=True, fmt="d", cmap="YlGnBu", cbar=False, linewidth=.5)
#centering heatmap
sns.heatmap(df_crosstab, annot=True, fmt='d', cmap="YlGnBu", cbar=True, center=df_crosstab.loc[9,6])
#plotting a correlation matrix
sns.heatmap(df.corr())


#ex.1
# Create a crosstab table of the data
pd_crosstab = pd.crosstab(df["Group"], df["YEAR"])
print(pd_crosstab)
# Plot a heatmap of the table
sns.heatmap(pd_crosstab)
# Rotate tick marks for visibility
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()


#ex.2
# Create the crosstab DataFrame
pd_crosstab = pd.crosstab(df["Group"], df["YEAR"])
# Plot a heatmap of the table with no color bar and using the BuGn palette
sns.heatmap(pd_crosstab, cbar=False, cmap="BuGn", linewidths=.3)
# Rotate tick marks for visibility
plt.yticks(rotation=0)
plt.xticks(rotation=90)
#Show the plot
plt.show()
plt.clf()


# =============================================================================
# Chapter 4
# =============================================================================
#using FacetGrid, factorplot and lmplot
#video 1
g=sns.FacetGrid(df, col="HIGHDEG")
g.map(sns.boxplot, "Tuition", order=["1", "2", "3", "4"])
 
#easier way
sns.factorplot(x="Tuition", data=df, col="HIGHDEG", kind="box")

g=sns.FacetGrid(df, col="HIGHDEG")
g.map(plt.scatter, 'Tuition', "SAT_AVG_ALL")

#lmplot
sns.lmplot(data=df, x="Tuition", y="SAY_AVG_ALL", col="HIGHDEG", fit_reg=False)
sns.lmplot(data=df, x="Tuition", y="SAY_AVG_ALL", col="HIGHDEG", row="Region")


#ex.1
# Create FacetGrid with Degree_Type and specify the order of the rows using row_order
g2 = sns.FacetGrid(df, 
             row="Degree_Type",
             row_order=['Graduate', 'Bachelors', 'Associates', 'Certificate'])
# Map a pointplot of SAT_AVG_ALL onto the grid
g2.map(sns.pointplot, 'SAT_AVG_ALL')
# Show the plot
plt.show()
plt.clf()


#ex.2

#part 1
# Create a factor plot that contains boxplots of Tuition values
sns.factorplot(data=df,
         x='Tuition',
         kind='box',
         row='Degree_Type')
plt.show()
plt.clf()

#part 2
# Create a facetted pointplot of Average SAT_AVG_ALL scores facetted by Degree Type 
sns.factorplot(data=df,
        x='SAT_AVG_ALL',
        kind='point',
        row='Degree_Type',
        row_order=['Graduate', 'Bachelors', 'Associates', 'Certificate'])
plt.show()
plt.clf()


#ex.3

#part 1
# Create a FacetGrid varying by column and columns ordered with the degree_order variable
g = sns.FacetGrid(df, col="Degree_Type", col_order=degree_ord)
# Map a scatter plot of Undergrad Population compared to PCTPELL
g.map(plt.scatter, 'UG', 'PCTPELL')
plt.show()
plt.clf()

#part 2
# Re-create the plot above as an lmplot
sns.lmplot(data=df,
        x='UG',
        y='PCTPELL',
        col="Degree_Type",
        col_order=degree_ord)
plt.show()
plt.clf()

#part 3
# Create an lmplot that has a column for Ownership, a row for Degree_Type and hue based on the WOMENONLY column
sns.lmplot(data=df,
        x='SAT_AVG_ALL',
        y='Tuition',
        col="Ownership",
        row='Degree_Type',
        row_order=['Graduate', 'Bachelors'],
        hue='WOMENONLY',
        col_order=inst_ord)
plt.show()
plt.clf()


#using PairGrid and pairplot
#video 2
g=sns.PairGrid(df, vars=["Fair_Mrkt_Rent", "Median_income"])
g=g.map(plt.scatter)

#customizing the PairGrid diagonals
g=sns.PairGrid(df, vars=["Fair_Mrkt_Rent", "Median_income"])
g=g.map_diag(plt.hist)
g=g.map_offdiag(plt.scatter)

sns.pairplot(df,vars=["Fair_Mrkt_Rent", "Median_income"], kind="reg", diag_kind="hist" )
sns.pairplot(df.query("BEDRMS<3"), vars=["Fair_Mrkt_Rent", "Median_income", "UTILITY"], hue="BEDRMS", palette="husl", plot_kws={'alpha':0.5} )


#ex.1

#part 1
# Create a PairGrid with a scatter plot for fatal_collisions and premiums
g = sns.PairGrid(df, vars=["fatal_collisions", "premiums"])
g2 = g.map(plt.scatter)
plt.show()
plt.clf()

#part 2
# Create the same pairgrid but map a histogram on the diag
g = sns.PairGrid(df, vars=["fatal_collisions", "premiums"])
g2 = g.map_diag(plt.hist)
g3 = g2.map_offdiag(plt.scatter)
plt.show()
plt.clf()


#ex.2

#part 1
# Create a pairwise plot of the variables using a scatter plot
sns.pairplot(data=df,
        vars=["fatal_collisions", "premiums"],
        kind='scatter')
plt.show()
plt.clf()

#part 2
# Plot the same data but use a different color palette and color code by Region
sns.pairplot(data=df,
        vars=["fatal_collisions", "premiums"],
        kind='scatter',
        hue='Region',
        palette='RdBu',
        diag_kws={'alpha':.5})
plt.show()
plt.clf()


#ex.3

#part 1
# Build a pairplot with different x and y variables
sns.pairplot(data=df,
        x_vars=["fatal_collisions_speeding", "fatal_collisions_alc"],
        y_vars=['premiums', 'insurance_losses'],
        kind='scatter',
        hue='Region',
        palette='husl')
plt.show()
plt.clf()

#part 2
# plot relationships between insurance_losses and premiums
sns.pairplot(data=df,
             vars=["insurance_losses", "premiums"],
             kind='reg',
             palette='BrBG',
              diag_kind= 'kde',
             hue='Region')
plt.show()
plt.clf()


#using JointGrid and jointplot
#video 3
g=sns.JointGrid(data=df, x="Tuition", y="ADM_RATE_ALL")
g.plot(sns.regplot, sns.distplot)

#advanced JointGrid
g=sns.JointGrid(data=df, x="Tuition", y="ADM_RATE_ALL")
g=g.plot_joint(sns.kdeplot)
g=g.plot_marginals(sns.kdeplot, shade=True)
g=g.annotate(stats.pearsonr)

sns.jointplot(data=df,  x="Tuition", y="ADM_RATE_ALL", kind="hex")

#customizing a jointplot
g=(sns.jointplot(x="Tuition", y="ADM_RATE_ALL", kind="scatter", xlim(0,25000), marginal_kws=dict(bins=15, rug=True), data=df.query('UG<2500 & Ownership=="Public"')).plot_joint(sns.kdeplot))


#ex.1

#part 1
# Build a JointGrid comparing humidity and total_rentals
sns.set_style("whitegrid")
g = sns.JointGrid(x="hum",
            y="total_rentals",
            data=df,
            xlim=(0.1, 1.0)) 
g.plot(sns.regplot, sns.distplot)
plt.show()
plt.clf()

#part 2
# Create a jointplot similar to the JointGrid 
sns.jointplot(x="hum",
        y="total_rentals",
        kind='reg',
        data=df)
plt.show()
plt.clf(


#ex.2

#part 1
# Plot temp vs. total_rentals as a regression plot
sns.jointplot(x="temp",
         y="total_rentals",
         kind='reg',
         data=df,
         order=2,
         xlim=(0, 1))
plt.show()
plt.clf()
        
#part 2
# Plot a jointplot showing the residuals
sns.jointplot(x="temp",
        y="total_rentals",
        kind='resid',
        data=df,
        order=2)
plt.show()
plt.clf()        
        

#ex.3

#part 1
# Create a jointplot of temp vs. casual riders
# Include a kdeplot over the scatter plot
g = (sns.jointplot(x="temp",
             y="casual",
             kind='scatter',
             data=df,
             marginal_kws=dict(bins=10, rug=True))
    .plot_joint(sns.kdeplot))
   
plt.show()
plt.clf()

#part 2
# Replicate the above plot but only for registered riders
g = (sns.jointplot(x="temp",
             y="registered",
             kind='scatter',
             data=df,
             marginal_kws=dict(bins=10, rug=True))
    .plot_joint(sns.kdeplot))
plt.show()
plt.clf()


#selecting seaborn plots
#video 4

# =============================================================================
# distplot() is best to start analysis (as rugplot and kdeplot)
# lmplot() is appropriate for regression analysis
# pairplot and jointplot is appropriate after performing the regression analysis
# 
# =============================================================================
