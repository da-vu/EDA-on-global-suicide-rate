'''
    ECE 143 Project
    Comprehensive Analysis on Suicide Rate around the World
    Authors: Payam Khorramshahi, Dan Vu, Dylan Perlson, Zhengdong Gao 
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def change_gdp_to_ints(df):
    '''convert gdp_for_year from string to ints 
    
       Args:
           df: dataframe

    '''

    gdp = []
    # convert gdp_for_year from string to ints 
    for i, string in enumerate(df[df.columns[8]]):
        gdp.append(int(string.replace(',','')))

    # put the results in a new column    
    gdp_frame =pd.DataFrame()
    gdp_frame['gdp'] = np.array(gdp)
    new_df = pd.concat((df,gdp_frame),axis=1)

    return new_df

def add_conts(df):
    '''adds continent column
    
        Args:
            df: dataframe
    
    '''
    cont = pd.read_csv('./countryContinent.csv',encoding = "ISO-8859-1")
    
    l = []
    count_list = []
    for country in df['country']:
        if country not in count_list:
            count_list.append(country)
            if country == 'Saint Vincent and Grenadines':
                country = 'Saint Vincent and the Grenadines'
            elif country == 'United Kingdom':
                country = 'United Kingdom of Great Britain and Northern Ireland'
            elif country == 'United States':
                country = 'United States of America'
            elif country == 'Macau':
                country = 'Macao'
            elif country == 'Republic of Korea':
                country = "Korea (Democratic People's Republic of)"
        
            l.append(cont[cont['country'] == country].continent.to_list()[0])

    cont_dict = {}         
    for i in range(len(count_list)):   
        cont_dict[count_list[i]] = l[i]


    new_col = pd.DataFrame()
    new_col['continent'] = df['country']
    new_col = new_col.replace({"continent": cont_dict})

    new_df = pd.concat((new_col, df), axis=1)

    return new_df


def remove_HDI(df):
    '''removes 'HDI for year' column from given df
    
       Args:
            df: dataframe

    '''
    new_df = df.drop(['HDI for year'], axis =1)
    new_df = new_df.dropna()
    return new_df

def rename_suicide_rate(df):
    '''rename 'suicides/100k pop' to 'suicides_per_100k'
        Args:
                df: dataframe
    '''
    new_df=df.rename(columns={"suicides/100k pop": "suicides_per_100k"})
    return new_df

def remove_2016(df):
    ''' removes rows from 2016
        Args:
                df: dataframe
    '''
    new_df=df[df.year != 2016]
    return new_df


def preprocess(df):
    '''prepares the dataframe
    
        Args:
            df: dataframe  
    '''
    new_df = remove_HDI(df)
    new_df = change_gdp_to_ints(new_df)
    new_df = rename_suicide_rate(new_df)
    new_df = remove_2016(new_df)
    new_df = add_conts(new_df)
    return new_df

def suiciderate_gender_year(df):
    '''plots suicide rate of each gender over the years
    
        Args:
            df: dataframe
    '''
    for gender in df['sex'].unique():
        df2 = df[df['sex'] == gender]
        dict = {}
        count =  np.array(df2['suicides_per_100k'])
        for ind, y in enumerate(df2['year']):
            if y not in dict:
                dict[y] = 0
                dict[y] += count[ind]
            else:
                dict[y] += count[ind]
                
        lists = list(dict.items())
        x,y = zip(*lists)
        plt.bar(x,y, label=gender)

        plt.legend(loc='best') 
    
    plt.title('Suicide Rate for each sex over different Years')
    plt.xlabel('Year')
    plt.ylabel('counts/100k')
    plt.grid(True)
    plt.show()

def suicidecount_gender_year(df):
    '''plots suicide count of each gender over the years
    
         Args:
            df: dataframe
    '''
    for gender in df['sex'].unique():
        df2 = df[df['sex'] == gender]
        dict = {}
        count =  np.array(df2['suicides_no'])
        for ind, y in enumerate(df2['year']):
            if y not in dict:
                dict[y] = 0
                dict[y] += count[ind]
            else:
                dict[y] += count[ind]
                
        lists = list(dict.items())
        x,y = zip(*lists)
        plt.bar(x,y, label=gender)

        plt.legend(loc='best') 
    
    plt.title('Suicide Counts for each sex over different Years')
    plt.xlabel('Year')
    plt.ylabel('counts')
    plt.grid(True)
    plt.show()

def suiciderate_gdp_gender(df, country='worldwide'):
    
    
    ''' visualizes suicide rate as a function of GDP seperately 
        for each gender.
    
        Args:
            df: dataframe
            country: country
    '''

    if country=='worldwide':
        p1 = df
    else:
        p1 = df[df['country'] == country]


    for gender in p1['sex'].unique():
        pm = p1[p1['sex'] == gender]
        sui_num = []
        gdp = []
        for year in pm['year'].unique():
            p2 = pm[pm['year'] == year]
            sui_num.append(sum(p2.suicides_per_100k))
            gdp.append(np.average(p2['gdp']))
            
        x = []
        y = []
        for i,j in sorted(zip(gdp,sui_num)):
            x.append(i)
            y.append(j)

        plt.scatter(x,y,label=gender)
    
    
    plt.legend(loc='best')
    
    plt.title('{}'.format(country))
    plt.xlabel('gdp')
    plt.ylabel('suicide count/100k')
    plt.grid()
    plt.show()



def suiciderate_age_year(df):
    
    '''visualizes sucide rate for each age group at different years.
       Args:
           df: dataframe
    '''

    for age in df['age'].unique():
        df2 = df[df['age'] == age]


        for gender in df['sex'].unique():
            dfm = df2[df2['sex']== gender]
            dict = {}
            count =  np.array(dfm['suicides_per_100k'])
            for ind, y in enumerate(dfm['year']):
                if y not in dict:
                    dict[y] = 0
                    dict[y] += count[ind]
                else:
                    dict[y] += count[ind]
                    
            
            lists = list(dict.items())
            x,y = zip(*lists)
            plt.bar(x,y, label=gender)
        
        
        plt.legend(loc='best')

        plt.title('age: {}'.format(age))
        plt.xlabel('year')
        plt.ylabel('suicides/100k pop')
        plt.grid()
        plt.show()

def suiciderate_age(df):
    
    '''visualizes suiciderate for each age without any other restriction
    
        Args:
            df: dataframe
    '''
    
    ages=[]
    rate=[]
    
    for age in df['age'].unique():
        ages.append(age)
        rate.append(np.sum(df[df['age']==age].suicides_per_100k))
        
    
    fig = plt.figure(figsize=(15,10))
    plt.pie(rate, labels=ages, autopct='%1.1f%%')
    plt.title('suicides rate population by age')
    plt.show()


def suiciderate_cont_time(df):
    
    '''visualizes the total number of suicide for each year.
        Args:
            df: dataframe
    '''
    
    for cont in df['continent'].unique():
        p1 = df[df['continent']==cont]
        dic ={}
        count = np.array(p1['suicides_per_100k'])
        for ind, y in enumerate(p1['year']):
            if y not in dic:
                dic[y]=0
                dic[y]+=count[ind]
            else:
                dic[y]+= count[ind]
                
                
        lists = list(dic.items())
        
        x,y=zip(*lists)
        
        plt.scatter(x,y,label=cont,marker='.')
        
    plt.legend(loc='best')
    
    plt.title('Suicide Rate by Continent of Different Years')
    plt.xlabel('Year')
    plt.ylabel('counts/100k population')
    plt.grid(True)
    plt.show()

def suiciderate_country_time(df,country_list):
    
    '''visualizes suicide rate at different countries
        
       Args:
           df: dataframe
           country_list: list of countries
    '''
    
    for country in country_list:
        p1 = df[df['country']==country]
        dic ={}
        count = np.array(p1['suicides_per_100k'])
        for ind, y in enumerate(p1['year']):
            if y not in dic:
                dic[y]=0
                dic[y]+=count[ind]
            else:
                dic[y]+= count[ind]
                
                
        lists = list(dic.items())
        
        x,y=zip(*lists)
        
        plt.scatter(x,y,label=country)
    
    plt.legend(loc='best')
    
    plt.title('Suicide Rate by Country of Different Years')
    plt.xlabel('Year')
    plt.ylabel('counts/100k population')
    plt.grid(True)
    plt.show()

def suicides_cont_avg(df):
    
    '''Visualizes the average suicide rate at different countries
        Args:
            df: dataframe
    '''
    
    for cont in df['continent'].unique():
        p1 = df[df['continent']==cont]
        cont_max_avg={cont:{'temp':0}}
        cont_min_avg={cont:{'temp':float('inf')}}
        for country in p1['country'].unique():
            p2 = p1[p1['country']==country]
            dic ={}
            count = np.array(p2['suicides_per_100k'])
            for ind, y in enumerate(p2['year']):
                if y not in dic:
                    dic[y]=0
                    dic[y]+=count[ind]
                else:
                    dic[y]+= count[ind]
                
            lists = np.array(list(dic.values()))
            
            if np.mean(lists)>list(cont_max_avg[cont].values())[0]:
                cont_max_avg={cont:{country:np.mean(lists)}}
            if np.mean(lists)<list(cont_min_avg[cont].values())[0] and np.mean(lists)>0:
                cont_min_avg={cont:{country:np.mean(lists)}}
                
        print('Continent: '+str(list(cont_max_avg.keys())[0])+'\n Highest rate: '+str(list(cont_max_avg.values()))+'\n Lowest rate: '+str(list(cont_min_avg.values())))



def suicide_gdp(df, *country):
    
    ''' Visualizes the impact of gdp for a list of countries on suicide rate in form of subplots
        Args:
            df: dataframe:
            country: series of countries (Must be greater than 1)
    '''
    
    assert len(country) % 2 ==0
    x = 2
    y = int(len(country)/2)
    w = 0
    
    fig, ax= plt.subplots(x,y, figsize=(10,8))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    for m in range(x):
        for n in range(y):
            
            p1 = df[df['country'] == country[w]]
            sui_num = []
            gdp = []
            for year in p1['year'].unique():
                p2 = p1[p1['year'] == year]
                sui_num.append(sum(p2.suicides_no))
                gdp.append(np.average(p2['gdp']))

            X = []
            Y = []
            for i,j in sorted(zip(gdp,sui_num)):
                X.append(i)
                Y.append(j)

            ax[m,n].scatter(X,Y)
            ax[m,n].set_title(str(country[w]))
            
            
            
            w+=1
    for ax in ax.flat:
        ax.set(xlabel='GDP', ylabel='Suicide Count')
        
    
    plt.show()


    
def barplot(frame, col_x, col_y, country):
    
    '''Creates a barplot for any two columns for a specific country:
        
        Args:
            frame: dataframe
            col_x: first column
            col_y: second column
            country: country
    '''
    
    reduced  = frame.where(frame['country'] == country)
    reduced = reduced.dropna()
    x_axis = reduced[col_x]
    y_axis = np.array(reduced[col_y])
    print('Number of data: ',len(y_axis))
    dict = {}
    for ind , val in enumerate(x_axis):
        if val not in dict:
            dict[val] = 0
            dict[val] += y_axis[ind]
        else:
            dict[val] += y_axis[ind]
    
    lists = list(dict.items())
    x,y = zip(*lists)
    x = list(x)
    y = list(y)
   

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(x,y)
    plt.title('{} vs. {} in {}'.format(col_y, col_x, country))
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.grid(True)
    plt.show()
    
    
def categorize(column):
    
    '''categorizes each string column by setting each value to a specific string using a hashing table 
        (dictionary) and applying to the column.
    
        Args:
            column: column
    '''
    
    dict = {}
    key = int(1)
    for i in np.unique(column):
        if i not in dict:
            dict[i] = key
            key+=1
    return dict

    
def suicide_gender_year(df):
    
    '''visualizes the suicide rate based on gender at different years>
        
        Args:
                df: dataframe
    '''
    
    df2 = df[df['sex'] == 'male']
    dict = {}
    count =  np.array(df2['suicides_per_100k'])
    for ind, y in enumerate(df2['year']):
        if y not in dict:
            dict[y] = 0
            dict[y] += count[ind]
        else:
            dict[y] += count[ind]
            
    
    lists = list(dict.items())
    x,y = zip(*lists)
    plt.bar(x,y, label='male')
    
    df2 = df[df['sex'] == 'female']
    dict = {}
    count =  np.array(df2['suicides_per_100k'])
    for ind, y in enumerate(df2['year']):
        if y not in dict:
            dict[y] = 0
            dict[y] += count[ind]
        else:
            dict[y] += count[ind]
            
    
    lists = list(dict.items())
    x,y = zip(*lists)
    plt.bar(x,y, label='female')
    
    
    plt.legend(loc='best')
    
    
    plt.title('Gender Suicide Rate over different Years')
    plt.xlabel('Year')
    plt.ylabel('counts')
    plt.grid(True)
    plt.show()
    
def suicides_gdp_gender(df, country):
    
    '''
       plot suicide counts versus gdp of each gender of a specific country
       
       Args:
           df: dataframe, country

    '''
    p1 = df[df['country'] == country]
    pm = p1[p1['sex'] == 'male']
    sui_num = []
    gdp = []
    for year in pm['year'].unique():
        p2 = pm[pm['year'] == year]
        sui_num.append(sum(p2.suicides_no))
        gdp.append(np.average(p2['gdp']))
        
    x = []
    y = []
    for i,j in sorted(zip(gdp,sui_num)):
        x.append(i)
        y.append(j)

    plt.scatter(x,y,label='male')
    
    pf = p1[p1['sex'] == 'female']
    sui_num = []
    gdp = []
    for year in pf['year'].unique():
        p2 = pf[pf['year'] == year]
        sui_num.append(sum(p2.suicides_no))
        gdp.append(np.average(p2['gdp']))
        
    x = []
    y = []
    for i,j in sorted(zip(gdp,sui_num)):
        x.append(i)
        y.append(j)
    
    plt.scatter(x,y,label='female')
    
    plt.legend(loc='best')
    
    
    plt.title('{}'.format(country))
    plt.xlabel('gdp')
    plt.ylabel('suicide count')
    plt.grid()
    plt.show()
    
    
def countries_suicide_rate(df):
    
    '''
       plot suicide counts by different countries
       
       Args:
           df: dataframe

    '''
    
    rate=[]
    countries=[]

    for country in df['country'].unique():
        countries.append(country)
        rate.append(np.sum(df[df['country']==country].suicides_per_100k))
        
    fig = plt.figure(figsize=(150,6))
    plt.bar(countries,rate)
    plt.show()
    

    
def suicides_age_year(df, age):
    
    '''
       plot suicide counts of each gender over years of a specific age period
       From 1985 to 2015
       
       Args:
           df: dataframe, age period

    '''
    
    df2 = df[df['age'] == age]
    dfm = df2[df2['sex']== 'male']
    dict = {}
    count =  np.array(dfm['suicides_per_100k'])
    for ind, y in enumerate(dfm['year']):
        if y not in dict:
            dict[y] = 0
            dict[y] += count[ind]
        else:
            dict[y] += count[ind]
            
    
    lists = list(dict.items())
    x,y = zip(*lists)
    plt.bar(x,y, label='male')
    
    
    
    dff = df2[df2['sex']=='female']
    
    dict = {}
    count =  np.array(dff['suicides_per_100k'])
    for ind, y in enumerate(dff['year']):
        if y not in dict:
            dict[y] = 0
            dict[y] += count[ind]
        else:
            dict[y] += count[ind]
            
    
    lists = list(dict.items())
    x,y = zip(*lists)
    plt.bar(x,y,label='female')
    
    plt.legend(loc='best')

    
    plt.title('age: {}'.format(age))
    plt.xlabel('year')
    plt.ylabel('suicides/100k pop')
    plt.grid()
    plt.show()


def suicides_per100k_age(df):
    
    '''
       plot pie graph to show suicide counts per 100k people ratio of different age periods
       
       Args:
           df: dataframe

    '''
    
    ages=[]
    rate=[]
    
    for age in df['age'].unique():
        ages.append(age)
        rate.append(np.sum(df[df['age']==age].suicides_per_100k))
        
    
    fig = plt.figure(figsize=(15,10))
    plt.pie(rate, labels=ages, autopct='%1.1f%%')
    plt.show()
    
def suicides_no_age(df):
    
    '''
       plot pie graph to show suicide counts ratio of different age periods
       
       Args:
           df: dataframe

    '''
    
    ages=[]
    rate=[]
    
    for age in df['age'].unique():
        ages.append(age)
        rate.append(np.sum(df[df['age']==age].suicides_no))
        
    
    fig = plt.figure(figsize=(15,10))
    plt.pie(rate, labels=ages, autopct='%1.1f%%')
    plt.show()
    
    
def suicide_by_age_pie(df):
    
    '''
       plot pie graph to show suicide counts ratio of different genders
       
       Args:
           df: dataframe

    '''
    
    aggregation_functions = {'suicides_no': 'sum'}
    df_sex = df.groupby(df['sex']).aggregate(aggregation_functions)
    plot = df_sex.plot.pie(subplots=True ,figsize=(5, 5),autopct='%1.1f%%',fontsize='17')
    plt.ylabel("")
    plt.title("Suicide percentages by sex",fontsize="17")
    
    
    
def suicide_by_year(df):
    
    '''
       plot suicide counts over years
       From 1985 to 2015
       
       Args:
           df: dataframe

    '''
    
    aggregation_functions = {'suicides_no': 'sum'}

    df_yr_sex = df.groupby(['year', 'sex']).agg(aggregation_functions)
    df_yr = df.groupby(df['year']).aggregate(aggregation_functions)
    df_yr = df_yr.drop(2016)
    df_yr.plot(kind="bar",rot = '50',figsize=(10,5),fontsize='14')
    plt.title('Number of Suicides per Year',fontsize='14')
    
    
def suicide_by_age_range(df):
    
    '''
       plot suicide counts versus different age periods of each gender 
       
       Args:
           df: dataframe

    '''
    
    
    aggregation_functions = {'suicides_no': 'sum'}

    
    df_age = df.groupby(df['age']).aggregate(aggregation_functions)
    df_age = df_age.sort_index()
    df_sex = df.groupby(df['sex']).aggregate(aggregation_functions)
    df_multiple = df.groupby(['age', 'sex']).agg(aggregation_functions)
    df_multiple = df_multiple.reset_index()
    df_age = df_age.rename(index={'5-14 years': '05-14 years'})
    df_age = df_age.sort_index()
    
    
    list_men, list_women = [],[]
    for i in range(len(df_multiple)):
        if i%2:
            list_men.append(df_multiple.values[i][2])
        else:
            list_women.append(df_multiple.values[i][2])

    plotdata = pd.DataFrame({
        "Men":list_men,
        "Women":list_women,

        }, 
        index=df_age.index.tolist()
    )

    plotdata = plotdata.rename(index={'5-14 years': '05-14 years'})
    plotdata = plotdata.rename(index={'05-14 years': '35-54 years','35-54 years': '05-14 years'})
    plotdata = plotdata.sort_index()
    plotdata.plot(kind="bar",rot=15,fontsize="14")
    plt.title("Suicides by age and sex",fontsize='14')
    plt.xlabel("Age",fontsize='14')
    plt.ylabel("Number of Suicides",fontsize='14')
    plt.show()

def suicide_by_sex(df):
    
    '''
       plot suicide counts versus different genders
       
       Args:
           df: dataframe

    '''
    
    aggregation_functions = {'suicides_no': 'sum'}
    df_sex = df.groupby(df['sex']).aggregate(aggregation_functions)
    ax = df_sex.plot.bar(rot=0)
    plt.title("Suicide Numbers by sex")
    plt.show()
    
    
    
def suicide_by_age(df):
    
    '''
       plot suicide counts versus different age periods of each genders
       
       Args:
           df: dataframe

    '''
    
    aggregation_functions = {'suicides_no': 'sum'}
    df_age = df.groupby(df['age']).aggregate(aggregation_functions)
    df_age = df_age.sort_index()
    df_sex = df.groupby(df['sex']).aggregate(aggregation_functions)
    df_multiple = df.groupby(['age', 'sex']).agg(aggregation_functions)
    df_multiple = df_multiple.reset_index()
    df_age = df_age.rename(index={'5-14 years': '05-14 years'})
    df_age = df_age.sort_index()
    ax = df_age.plot.bar(rot=15,fontsize='14')

    plt.title("Suicides by Age",fontsize='14')
    plt.xlabel("Age",fontsize='14')
    plt.ylabel("Number of Suicides",fontsize='14')
    plt.show()
    


def suicide_pearson_population(df):
    
    '''
       calculate pearson correlation between suicides counts and the population of each country 
       years 1985 1986 2016 are removed
       
       Args:
           df: dataframe

    '''
    
    country_list=np.unique(df['country'])
    year_list=list(np.unique(df['year']))

    year_list.remove(1985)
    year_list.remove(1986)
    year_list.remove(2016)
        
    pear=[]
    
    for co in country_list:
        
        df_c = df[df['country'] == co]
       
        dff_facto=[]
        dff_sui=[]
        
        for year in year_list:
            
            df_y = df_c[df_c['year'] == year]
            df_y_s=list(df_y['suicides_no'])
            n=sum(df_y_s)
            
            dff_sui.append(n)
            
            df_y_f=list(df_y['population'])
            df_y_f_int=[]
            
            df_y_f_int=df_y_f
                        
            nn=sum(df_y_f_int)
     
            dff_facto.append(nn)            
        
        cor=np.corrcoef(dff_sui,dff_facto)
        pear.append(cor[1][0])

    return pear

def Sui_by_pop(df, country):
    
    '''
       plots suicide counts of each country versus the population
       years 1985 1986 2016 are removed
       
       Args:
           df: dataframe, country

    '''
    
    df_c = df[df['country'] == country]
    country_list=np.unique(df['country'])
    year_list=list(np.unique(df['year']))

    year_list.remove(1985)
    year_list.remove(1986)
    year_list.remove(2016)

    dff_population=[]
    dff_sui=[]

    for year in year_list:

        df_y = df_c[df_c['year'] == year]
        df_y_s=list(df_y['suicides_no'])
        n=sum(df_y_s)

        dff_sui.append(n)

        df_y_f=list(df_y['population'])
        df_y_f_int=[]


        df_y_f_int=df_y_f


        nn=sum(df_y_f_int)

        dff_population.append(nn)       

    fig = plt.figure(figsize=(8,5))

    plt.scatter(dff_population,dff_sui)
    plt.title('{}'.format(country) ,Fontsize=25)
    plt.tick_params(labelsize=15)   
    plt.xticks(rotation=15);
    plt.xlabel('Population',Fontsize=25)
    plt.ylabel('num of suicide',Fontsize=25)
