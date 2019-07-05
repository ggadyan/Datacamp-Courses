# =============================================================================
# Chapter 1
# =============================================================================
import matplotlib.pyplot as plt 
fig, ax = plt.subplots()
ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"], marker="o", linestyle="--",color="b")
ax.plot(austin_weather["MONTH"], austin_weather["MLY-PRCP-NORMAL"], marker="v", linestyle="--",color="r")

#customizing axis label and title
ax.set_xlabel("Time(months)")
ax.set_ylabel("Average temperature (Fahrenheit degrees)")
ax.set_title("Weather patterns in Austin and Seattle")
plt.show()

#for marker stypes use : "https://matplotlib.org/api/markers_api.html"
#for linestyles use: "https://matplotlib.org/gallery/lines_bars_and_markers/line_styles_reference.html"

#Small multiples
fig, ax = plt.subplots(2,1, sharey=True) #grid of subplots with shared y axis
ax[0].plot(seattle_weather["MONTH"], seattle_weather["MLY-PRCP-NORMAL"], color="b")
ax[0].plot(seattle_weather["MONTH"], seattle_weather["MLY-PRCP-25PCTL"],  linestyle="--",color="b")
ax[0].plot(seattle_weather["MONTH"], seattle_weather["MLY-PRCP-75PCTL"],  linestyle="--",color="b")

ax[1].plot(austin_weather["MONTH"], austin_weather["MLY-PRCP-NORMAL"], color="r")
ax[1].plot(austin_weather["MONTH"], austin_weather["MLY-PRCP-25PCTL"],  linestyle="--",color="r")
ax[1].plot(austin_weather["MONTH"], austin_weather["MLY-PRCP-75PCTL"],  linestyle="--",color="r")


ax[0].set_ylabel( "Precipitation (inches)")
ax[1].set_ylabel( "Precipitation (inches)")

ax[1].set_xlabel(" Time (months) ")
plt.show()

# =============================================================================
# Chapter 2
# =============================================================================
#read data with time index
import matplotlib.pyplot as plt 
fig, ax = plt.subplots()
sixties=climate_change["1960-01-01":"1969-12-31"]
ax.plot(sixties.index, sixties["co2"])
ax.set_xlabel("Time")
ax.set_ylabel ("CO2 (ppm)")
plt.show()


import pandas as pd
# Read the data from file using read_csv
climate_change = pd.read_csv('climate_change.csv', parse_dates=["date"], index_col="date")


#plotting time-series with different variables
fig, ax = plt.subplots()
ax.plot(climate_change.index, climate_change["co2"], color="blue")
ax.set_xlabel("Time")
ax.set_ylabel ("CO2 (ppm)", color="blue")
ax.tick_params("y", colors="blue")
ax2=ax.twinx()
ax2.plot(climate_change.index, climate_change["relative_temp"], color="red")
ax2.set_ylabel("Relative temperature (Celsius) ", color="red")
ax2.tick_params("y", colors="red")
plt.show()

#a function that plots time-series
def plot_timeseries(axes, x, y, color, xlabel, ylabel):
    axes.plot(x,y,color=color)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel, color=color)
    axes.tick_params("y", colors=color)

fig, ax = plt.subplots()

# Plot the CO2 levels time-series in blue
plot_timeseries(ax, climate_change.index, climate_change["co2"], "blue", "Time (years)", "CO2 levels")
# Create a twin Axes object that shares the x-axis
ax2=ax.twinx()
# Plot the relative temperature data in red
plot_timeseries(ax, climate_change.index, climate_change["relative_temp"], "red", "Time (years)", "Relative temperature (Celsius)")
plt.show()


#annotating time-series data

fig, ax = plt.subplots()
plot_timeseries(ax, climate_change.index, climate_change["co2"], "blue", "Time (years)", "CO2 levels")
ax2=ax.twinx()
plot_timeseries(ax, climate_change.index, climate_change["relative_temp"], "red", "Time (years)", "Relative temperature (Celsius)")
ax2.annotate(">1 degree", xy=(pd.Timestamp("2015-10-06"),1), xytext=(pd.Timestamp("2008-10-06"), -0.2), arrowprops={"arrowstyle":"->", "color":"gray"})
#customizing annotations at "https://matplotlib.org/users/annotations.html"
#ex1
fig, ax = plt.subplots()
# Plot the relative temperature data
ax.plot(climate_change.index, climate_change["relative_temp"])
# Annotate the date at which temperatures exceeded 1 degree
ax.annotate(">1 degree", xy=(pd.Timestamp("2015-10-06"),1), xytext=(pd.Timestamp("2008-10-06"), -0.2))
plt.show()

#ex2
fig, ax = plt.subplots()
# Plot the CO2 levels time-series in blue
plot_timeseries(ax, climate_change.index, climate_change["co2"], "blue", "Time (years)", "CO2 levels")
# Create an Axes object that shares the x-axis
ax2 = ax2=ax.twinx()
# Plot the relative temperature data in red
plot_timeseries(ax, climate_change.index, climate_change["relative_temp"], "red", "Time (years)", "Relative temp (Celsius)")
# Annotate point with relative temperature >1 degree
ax2.annotate(">1 degree", xy=(pd.Timestamp("2015-10-06"),1), xytext=(pd.Timestamp("2008-10-06"), -0.2), arrowprops={"arrowstyle":"->", "color":"gray"})
plt.show()

# =============================================================================
# Chapter 3
# =============================================================================
#quantatitive comparisons: bar-charts
#ex.1
fig, ax = plt.subplots()
ax.bar(medals.index, medals["Gold"], label="Gold")
ax.bar(medals.index, medals["Silver"], bottom=medals["Gold"], label="Silver")
ax.bar(medals.index, medals["Bronze"], bottom=medals["Gold"]+medals["Silver"], label="Bronze")
ax.set_xticklabels(medals.index, rotation=90)
ax.set_ylabel("Number of medals")
ax.legend
plt.show()


#quantitative comparisons:histogram

#fig,ax=plt.subplots()
#ax.bar("Rowing", mens_rowing["Height"].mean())
#ax.bar("Gymnastics", mens_gymnastic["Height"]. mean())
#ax.set_ylabel("Height (cm)")
#plt.show()

#video
fig,ax=plt.subplots()
ax.hist(mens_rowing["Height"], label="Rowing", bins=5)
#ax.hist(mens_rowing["Height"], label="Rowing", bins=[150,160,170,180,190,200,210], histtype="step")
ax.hist(mens_gymnastic["Height"], label="Gymnastics", bins=5)
#ax.hist(mens_gymnastic"Height"], label="Gymnastics", bins=[150,160,170,180,190,200,210], histtype="step")
ax.set_xlabel("Height (cm)")
ax.set_ylabel("# of observations")
ax.legend()
plt.show()

#ex.1
fig, ax = plt.subplots()
# Plot a histogram of "Weight" for mens_rowing
ax.hist(mens_rowing["Weight"])
# Compare to histogram of "Weight" for mens_gymnastics
ax.hist(mens_gymnastics["Weight"])
# Set the x-axis label to "Weight (kg)"
ax.set_xlabel("Weight (kg)")
# Set the y-axis label to "# of observations"
ax.set_ylabel("# of observations")
plt.show()

#ex.2
fig, ax = plt.subplots()
# Plot a histogram of "Weight" for mens_rowing
ax.hist(mens_rowing["Weight"], label="Rowing", bins=5, histtype="step")
# Compare to histogram of "Weight" for mens_gymnastics
ax.hist(mens_gymnastics["Weight"], label="Gymnastics", bins=5, histtype="step")
ax.set_xlabel("Weight (kg)")
ax.set_ylabel("# of observations")
# Add the legend and show the Figure
ax.legend()
plt.show()

#video
#statistical plotting
fix, ax=plt.subplots()
ax.bar("Rowing", mens_rowing["Height"].mean(), yerr=mens_rowing["Height"].std())
ax.bar("Gymnastics", mens_gymnastics["Height"].mean(), yerr=mens_gymnastics["Height"].std())
ax.set_ylabel("Height (cm)")
plt.show()


#adding error bars to plots
fig, ax = plt.subplots()
ax.errorbar(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"], yerr=seattle_weather["MLY-TAVG-STDDEV"] )
ax.errorbar(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"], yerr=austin_weather["MLY-TAVG-STDDEV"] )
ax.set_ylabel("Temperature (Fahrenheit)")
plt.show()


#adding boxplots
fig,ax=plt.subplots()
ax.boxplot([mens_rowing["Height"], mens_gymnastics["Height"]])
ax.set_xticklabels(["Rowing", "Gymnastics"])
ax.set_ylabel("Height (cm)")
plt.show()

#ex.1
fig, ax = plt.subplots()
# Add a bar for the rowing "Height" column mean/std
ax.bar("Rowing", mens_rowing["Height"].mean(), yerr=mens_rowing["Height"].std())
# Add a bar for the gymnastics "Height" column mean/std
ax.bar("Gymnastics", mens_gymnastics["Height"].mean(), yerr=mens_gymnastics["Height"].std())
# Label the y-axis
ax.set_ylabel("Height (cm)")
plt.show()

#ex.2
fig, ax = plt.subplots()
# Add Seattle temperature data in each month with error bars
ax.errorbar(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"], yerr=seattle_weather["MLY-TAVG-STDDEV"] )
# Add Austin temperature data in each month with error bars
ax.errorbar(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"], yerr=austin_weather["MLY-TAVG-STDDEV"] )
# Set the y-axis label
ax.set_ylabel("Temperature (Fahrenheit)")
plt.show()

#ex.3
fig, ax = plt.subplots()
# Add a boxplot for the "Height" column in the DataFrames
ax.boxplot([mens_rowing["Height"], mens_gymnastics["Height"]])
# Add x-axis tick labels:
ax.set_xticklabels(["Rowing", "Gymnastics"])
# Add a y-axis label
ax.set_ylabel("Height (cm)")
plt.show()


#quantitative comparisons: scatter plots
#video

fig, ax=plt.subplots()
ax.scatter(climate_change["co2"], climate_change["relative_temp"])
ax.set_xlabel("CO2 (ppm)")
ax.set_ylabel("Relative temperature (Celsius)")
plt.show()

#customizing scatter plots
eighties=climate_change["1980-01-01":"1989-12-31"]
nineties=climate_change["1990-01-01":"1999-12-31"]
fig,ax=plt.subplots()
ax.scatter(eighties["co2"], eighties["relative_temp"],color="red", label="eighties")
ax.scatter(nineties["co2"], nineties["relative_temp"],color="blue", label="nineties")
ax.legend()
ax.set_xlabel("CO2 (ppm)")
ax.set_ylabel("Relative temperature (Celsius)")
plt.show()

#encoding a third variable by color
fig,ax=plt.subplots()
ax.scatter (climate_change["co2"], climate_change["relative_temp"], c=climate_change.index)
ax.set_xlabel("CO2 (ppm)")
ax.set_ylabel("Relative temperature (Celsius)")
plt.show()

#ex.1
fig, ax = plt.subplots()
# Add data: "co2" on x-axis, "relative_temp" on y-axis
ax.scatter(climate_change["co2"], climate_change["relative_temp"])
# Set the x-axis label to "CO2 (ppm)"
ax.set_xlabel("CO2 (ppm)")
# Set the y-axis label to "Relative temperature (C)"
ax.set_ylabel("Relative temperature (C)")
plt.show()

#ex.2
fig, ax = plt.subplots()
# Add data: "co2", "relative_temp" as x-y, index as color
ax.scatter (climate_change["co2"], climate_change["relative_temp"], c=climate_change.index)
# Set the x-axis label to "CO2 (ppm)"
ax.set_xlabel("CO2 (ppm)")
# Set the y-axis label to "Relative temperature (C)"
ax.set_ylabel("Relative temperature (C)")
plt.show()


# =============================================================================
# Chapter 4
# =============================================================================


#preparing your figures to share with others
#video
plt.style.use("ggplot")
#plt.style.use("seaborn-colorblind") #or "tableu-colorblind10"
#plt.style.use("bmh")
#plt.style.use("grayscale")
fig, ax = plt.subplots()
ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"] )
ax.plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"])
ax.set_xlabel("Time (months)")
ax.set_ylabel("Average temperature (Fahrenheit degrees)")
plt.show()
#going back to the default style
plt.style.use("default")
# find more styles at: "https://matplotlib.org/gallery/style_sheets/style_sheets_reference"

#multiple
#grayscale style

#ex.1
#a.
# Use the "ggplot" style and create new Figure/Axes
plt.style.use("ggplot")
fig, ax = plt.subplots()
ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"])
plt.show()
#b.
# Use the "Solarize_Light2" style and create new Figure/Axes
plt.style.use("Solarize_Light2")
fig, ax = plt.subplots()
ax.plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"])
plt.show()


#saving your visualizations/sharing your visualizations with others
#video

fig,ax=plt.subplots()
ax.bar(medals.index, medals["Gold"])
ax.set_xticklabels(medals.index, rotation=90)
ax.set_ylabel("Number of medals")
fig.savefig("gold-medals.png")
#fig.savefig("gold_medals.jpg", quality=50) #number between 1 and 100, preferably less than 95
#fig.savefig("gold_models.svg")
#fig.savefig("gold_medals.png", dpi=300) #the higher the number the more densely the image will be rendered
ls #a listing of the files in the present working directory
fig.set_size_inches([5,3])

#ex.1
#1
plt.show()
#2
fig.savefig("my_figure.png")
#3
fig.savefig("my_figure_300dpi.png", dpi=300)


#ex.2
#1
# Set figure dimensions and save as a PNG
fig.set_size_inches([3,5])
fig.savefig("figure_3_5.png")
#2
# Set figure dimensions and save as a PNG
fig.set_size_inches([5,3])
fig.savefig("figure_5_3.png")


#automating figures from data
#video

# =============================================================================
# why automate?
# -ease and speed
# -flexibility
# -robustness
# -reproducibility 
# =============================================================================

sports=summer_2016_medals["Sport"].unique()
print(sports)

fig,ax=plt.subplots()
for sport in sports:
    sport_df=summer_2016_medals[summer_2016_medals["Sport"]==sport]
    ax.bar(sport, sport_df["Height"].mean(), yerr=sport_df["Height"].std())
ax.set_ylabel("Height (cm)")
ax.set_xticklabels(sports, rotation=90)
plt.show()

#ex.1
# Extract the "Sport" column
sports_column = summer_2016_medals["Sport"]
# Find the unique values of the "Sport" column
sports =summer_2016_medals["Sport"].unique()
# Print out the unique sports values
print(sports)



#ex.2
fig, ax = plt.subplots()
# Loop over the different sports branches
for sport in sports:
  # Extract the rows only for this sport
  sport_df=summer_2016_medals[summer_2016_medals["Sport"]==sport]
  # Add a bar for the "Weight" mean with std y error bar
  ax.bar(sport, sport_df["Weight"].mean(), yerr=sport_df["Weight"].std())
ax.set_ylabel("Height (cm)")
ax.set_xticklabels(sports, rotation=90)
# Save the figure to file
fig.savefig("sports_weights.png")

#where to go next
#video
#https://matplotlib.org/gallery
#https://scitools.org.uk/cartopy/docs/latest/
#http://seaborn.pydata.org/examples/index










