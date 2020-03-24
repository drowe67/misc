#!/usr/bin/python3
# Most of this code came from Mohammad Ashhad - Thanks:
#   https://towardsdatascience.com/analyzing-coronavirus-covid-19-data-using-pandas-and-plotly-2e34fe2c4edc
#
# Some plots at the end by David Rowe 2020
#
# usage:
#   $ ./covid19.py

## Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
#%matplotlib inline
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
plt.rcParams['figure.figsize'] = [15, 5]
from IPython import display
from ipywidgets import interact, widgets
from datetime import date

## Read Data for Cases, Deaths and Recoveries
ConfirmedCases_raw=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
#Deaths_raw=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')
#Recoveries_raw=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')
### Melt the dateframe into the right shape and set index
def cleandata(df_raw):
    df_cleaned=df_raw.melt(id_vars=['Province/State','Country/Region','Lat','Long'],value_name='Cases',var_name='Date')
    df_cleaned=df_cleaned.set_index(['Country/Region','Province/State','Date'])
    return df_cleaned 

# Clean all datasets
ConfirmedCases=cleandata(ConfirmedCases_raw)
#Deaths=cleandata(Deaths_raw)
#Recoveries=cleandata(Recoveries_raw)

### Get Countrywise Data
def countrydata(df_cleaned,oldname,newname):
    df_country=df_cleaned.groupby(['Country/Region','Date'])['Cases'].sum().reset_index()
    df_country=df_country.set_index(['Country/Region','Date'])
    df_country.index=df_country.index.set_levels([df_country.index.levels[0], pd.to_datetime(df_country.index.levels[1])])
    df_country=df_country.sort_values(['Country/Region','Date'],ascending=True)
    df_country=df_country.rename(columns={oldname:newname})
    return df_country
  
ConfirmedCasesCountry=countrydata(ConfirmedCases,'Cases','Total Confirmed Cases')
#DeathsCountry=countrydata(Deaths,'Cases','Total Deaths')
#RecoveriesCountry=countrydata(Recoveries,'Cases','Total Recoveries')

### Get DailyData from Cumulative sum
def dailydata(dfcountry,oldname,newname):
    dfcountrydaily=dfcountry.groupby(level=0).diff().fillna(0)
    dfcountrydaily=dfcountrydaily.rename(columns={oldname:newname})
    return dfcountrydaily

NewCasesCountry=dailydata(ConfirmedCasesCountry,'Total Confirmed Cases','Daily New Cases')
#NewDeathsCountry=dailydata(DeathsCountry,'Total Deaths','Daily New Deaths')
#NewRecoveriesCountry=dailydata(RecoveriesCountry,'Total Recoveries','Daily New Recoveries')

CountryConsolidated=pd.merge(ConfirmedCasesCountry,NewCasesCountry,how='left',left_index=True,right_index=True)
#CountryConsolidated=pd.merge(CountryConsolidated,NewDeathsCountry,how='left',left_index=True,right_index=True)
#CountryConsolidated=pd.merge(CountryConsolidated,DeathsCountry,how='left',left_index=True,right_index=True)
#CountryConsolidated=pd.merge(CountryConsolidated,RecoveriesCountry,how='left',left_index=True,right_index=True)
#CountryConsolidated=pd.merge(CountryConsolidated,NewRecoveriesCountry,how='left',left_index=True,right_index=True)
#CountryConsolidated['Active Cases']=CountryConsolidated['Total Confirmed Cases']-CountryConsolidated['Total Deaths']-CountryConsolidated['Total Recoveries']
#CountryConsolidated['Share of Recoveries - Closed Cases']=np.round(CountryConsolidated['Total Recoveries']/(CountryConsolidated['Total Recoveries']+CountryConsolidated['Total Deaths']),2)
#CountryConsolidated['Death to Cases Ratio']=np.round(CountryConsolidated['Total Deaths']/CountryConsolidated['Total Confirmed Cases'],3)
CountryConsolidated.tail(2)

#TotalCasesCountry=CountryConsolidated.max(level=0)['Total Confirmed Cases'].reset_index().set_index('Country/Region')
#TotalCasesCountry=TotalCasesCountry.sort_values(by='Total Confirmed Cases',ascending=False)
#TotalCasesCountryexclChina=TotalCasesCountry[~TotalCasesCountry.index.isin(['Mainland China','Others'])]
#Top10countriesbycasesexclChina=TotalCasesCountryexclChina.head(10)
#TotalCasesCountrytop10=TotalCasesCountry.head(10)
#fig = go.Figure(go.Bar(x=Top10countriesbycasesexclChina.index, y=Top10countriesbycasesexclChina['Total Confirmed Cases'],
#                      text=Top10countriesbycasesexclChina['Total Confirmed Cases'],
#            textposition='outside'))
#fig.update_layout(title_text='Top 10 Countries by Total Confirmed Cases Excluding China')
#fig.update_yaxes(showticklabels=False)

ItalyFirstCase=CountryConsolidated.loc['Italy']['Total Confirmed Cases'].reset_index().set_index('Date')
AustraliaFirstCase=CountryConsolidated.loc['Australia']['Total Confirmed Cases'].reset_index().set_index('Date')
USFirstCase=CountryConsolidated.loc['US']['Total Confirmed Cases'].reset_index().set_index('Date')
SpainFirstCase=CountryConsolidated.loc['Spain']['Total Confirmed Cases'].reset_index().set_index('Date')

ItalyGrowth=ItalyFirstCase[ItalyFirstCase.ne(0)].dropna().reset_index()
AustraliaGrowth=AustraliaFirstCase[AustraliaFirstCase.ne(0)].dropna().reset_index()
USGrowth=USFirstCase[USFirstCase.ne(0)].dropna().reset_index()
SpainGrowth=SpainFirstCase[SpainFirstCase.ne(0)].dropna().reset_index()

# Plotting stuff ----------------------------------------------------

# this is moving fast, and early data is noisey, we just want the last few days
last_days = 14

def plot_cases(country, sub_plot, data):
    x = np.arange(1,last_days+1)
    y = np.array(data['Total Confirmed Cases'])[-last_days:]
    plt.subplot(sub_plot); plt.plot(x,y); plt.title(country + ' Cases');
    plt.grid()

# a linear slope on a log plot means exponential growth which is bad
def plot_logcases(country, data):
    x = np.arange(1,last_days+1)
    y = np.array(data['Total Confirmed Cases'])[-last_days:]
    plt.semilogy(x,y,label=country);
    plt.legend()
    
# N(t)=N(0)*2**(t/Td)
# Td = t/log2(N(t)/N(0)
def plot_doubling(country, data):
    # estimate over a window of 3 days so not too noisey
    t = 3
    cases = np.array(data['Total Confirmed Cases'], dtype=float)[-last_days-t:]
    ratio = np.log2(cases[t:]/cases[0:-t])
    # do something sensible just in case we get a 0 ratio
    ratio[ratio  == 0] = 1000
    Td = t/ratio
    x = np.arange(1,ratio.size+1)
    plt.plot(x,Td,label=country);
    plt.legend()
    
today = date.today()

plt.figure(1,figsize=(8,6))
plt.tight_layout()
plot_cases("Australia", "221", AustraliaGrowth)
plot_cases("US", "222", USGrowth)
plot_cases("Italy", "223", ItalyGrowth)
plot_cases("Spain", "224", SpainGrowth)
plt.suptitle('Total Cases ' + today.strftime("%B %d, %Y"))
plt.savefig("cases.png")

plt.figure(2,figsize=(8,6))
plot_logcases("Australia", AustraliaGrowth)
plot_logcases("US", USGrowth)
plot_logcases("Italy", ItalyGrowth)
plot_logcases("Spain", SpainGrowth)
plt.title('Total Cases ' + today.strftime("%B %d, %Y"))
plt.grid()
plt.savefig("logcases.png")

plt.figure(3,figsize=(8,6))
plot_doubling("Australia", AustraliaGrowth)
plot_doubling("US", USGrowth)
plot_doubling("Italy", ItalyGrowth)
plot_doubling("Spain", SpainGrowth)
plt.ylabel("Doubling Time (days)");
plt.title('Doubling Time ' + today.strftime("%B %d, %Y"))
plt.grid()
plt.savefig("doublingtime.png")
plt.show()

