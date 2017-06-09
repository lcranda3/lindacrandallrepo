#!/usr/bin/python
# Analyze the cernstaff_mod.csv file with numpy, pandas and matplotlib.
# The file contains one row per employee, detailing:
# Category, Flag, Age, YearsInService, Nunmber of children, Grade, Step, Hours per week, Cost (in Swiss francs), Division, Nationality, ChildrenAges

# The idea is to learn how to read in and manipulate a table of data.
# Try to do as many of these as you can:
# 1) plot a table with the number of people as a function of nation and division
# 2) make a profile of the average cost per nation.
# 3) make a stack plot of some nations as a function of the grade
# and 4) make a histogram of the age distribution (choose an appropriate number of bins, and lower and upper limits).

# To continue with your understanding of this data. Answer the following questions:
# 5) How many british employees have 2 children?
# 6) What is the average cost of people over 50? And of people below 35? The units of cost are swiss francs.
# 7) Plot the age of all children. Why is the number of entries of this histogram different from the histogram of Age of employees or the employees cost?
# 8) Plot the age of the second child for employees whose cost is more than 10000.
# 9) Which nation has the highest grade on average? And which nation has the most children per capita?
# 10) Overlay the Age and ChildrenAges histograms on the same figure (use different colors and different line styles, add a legend)
# 11) Add the two histograms Age and ChildrenAges (and multiply ChildrenAges by 2).
# 12) Plot the ratio of Cost and Hrweek and use a log scale on the vertical axis. Can you do the ratio of Age and ChildrenAge?
# 13) Normalize the two histograms Age and ChildrenAge to the same area (say: 100) and plot them together (overlaid).
#     By normalizing them to the same area, we can focus on differences in their shape.
# 14) If you had to use only the Age and ChildreAge histograms (with 20 bins and without having access to the original table), would you be able to calculate how many employees over 50 have children under 10? Would you be able to calculate how many children are 21 years or older? What is the advantage  histograms?
# 15) Do your own investigations and report an interesting observation about the data.





def main(finpath='cernstaff_mod.csv'):
    '''  Small example how to read the file into a Pandas DataFrame and do some basic things.
         The same can be done directly with numpy instead of Pandas. '''
    import pandas as pd
    # Learn about Pandas DataFrames and how to read in tables from ascii files
    df = pd.read_csv(finpath, index_col=False, skiprows=1,
                     names=['Cat', 'Flag', 'Age', 'Srvc', 'Nch', 'Grade', 'Step', 'Hrweek', 'Cost', 'Div', 'Nation',
                            'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'], lineterminator='\n')
    # Let's merge all 10 columns of children ages into a new DataFrame column, indeed a list:
    df['ChildrenAges'] = list(zip(df.c0, df.c1, df.c2, df.c3, df.c4, df.c5, df.c6, df.c7, df.c8, df.c9))
    # Now we can apply the selection criteria we want to select subsamples:
    print df[(df['Age'] > 30) & (df['Nch'] > 3)][
        ['Cost']].values  # gives the cost values for all members older than 30 and with 3 children
    print df['ChildrenAges'].values
    # DataFrames have a ton of useful commands:
    print df['Age'].describe()
    # Cuts:
    # print df[df['Nation']=='GB'][['Age']].values
    # We can directly plot some of these
    # Learn more here: http://pandas.pydata.org/pandas-docs/version/0.19.1/visualization.html
    import matplotlib.pyplot as plt
    plt.hist(df[df['Nation'] == 'GB'].Age.values, bins=20)
    plt.show()  # just close the window to come back to the script

    df.plot(kind="scatter", x='Age', y='Cost', ylim=(0, 20000))
    # plt.show()
    plt.savefig('myplot.png')


# Execute the main program.
#main('cernstaff_mod.csv')

def question1(finpath='cernstaff_mod.csv'):
    import pandas as pd
    # Learn about Pandas DataFrames and how to read in tables from ascii files
    df = pd.read_csv(finpath, index_col=False, skiprows=1,
                     names=['Cat', 'Flag', 'Age', 'Srvc', 'Nch', 'Grade', 'Step', 'Hrweek', 'Cost', 'Div', 'Nation',
                            'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'], lineterminator='\n')
    # Let's merge all 10 columns of children ages into a new DataFrame column, indeed a list:
    df['ChildrenAges'] = list(zip(df.c0, df.c1, df.c2, df.c3, df.c4, df.c5, df.c6, df.c7, df.c8, df.c9))
    nationlist = []
    for item in df['Nation']:
        if item not in nationlist:
            nationlist.append(item)
    #print nationlist
    numnationlist=[]
    for item in nationlist:
        i=0
        for nation in df['Nation']:
            if nation==item:
                i+=1
        numnationlist.append(i)
    #print numnationlist
    divlist = []
    for item in df['Div']:
        if item not in divlist:
            divlist.append(item)
    #print divlist
    nationdivdict={}
    for nation in nationlist:
       for div in divlist:
            i=len(df[(df['Nation'] == nation) & (df['Div'] == div)])
            nationdivdict[str(nation)+" and "+str(div)] = i
    #print nationdivdict
    matrix=[]
    matrix2=[]
    for key in nationdivdict:
        matrix.append([key,nationdivdict[key]])
        matrix2.append([nationdivdict[key],key])
    #print matrix
    matrix.sort()
    #print matrix
    data_matrix=[['Nation and Division','Number of people']]+matrix
    #print data_matrix
    import matplotlib.pyplot as plt

    import numpy as np


    fig, ax = plt.subplots()
    i=1
    for nation in nationlist:
        j=1
        ax.text(i+.25,.25,nation)
        for div in divlist:
            ax.text(.25,j+.25,div)
            ax.text(1+i-.5,j+.5,len(df[(df['Nation'] == nation) & (df['Div'] == div)]), va='center', ha='center')
            j+=1
        i+=1
    ax.set_xlim([0, 16])
    ax.set_ylim([0, 14])
    ax.vlines(range(0,16,1),0,14)
    ax.hlines(range(0,16,1),0,16)
    ax.set_xlabel('Nations')
    ax.set_xticklabels("")
    ax.set_yticklabels("")
    ax.set_ylabel('Divisions')
    ax.set_title('Question 1: People as a function of Nation and Division')
    ax.grid()
    plt.savefig('question1.png')
    plt.show()

question1('cernstaff_mod.csv')

def question2(finpath='cernstaff_mod.csv'):
    '''  Small example how to read the file into a Pandas DataFrame and do some basic things.
         The same can be done directly with numpy instead of Pandas. '''
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # Learn about Pandas DataFrames and how to read in tables from ascii files
    df = pd.read_csv(finpath, index_col=False, skiprows=1,
                     names=['Cat', 'Flag', 'Age', 'Srvc', 'Nch', 'Grade', 'Step', 'Hrweek', 'Cost', 'Div', 'Nation',
                            'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'], lineterminator='\n')
    # Let's merge all 10 columns of children ages into a new DataFrame column, indeed a list:
    df['ChildrenAges'] = list(zip(df.c0, df.c1, df.c2, df.c3, df.c4, df.c5, df.c6, df.c7, df.c8, df.c9))
    nationlist = []
    for item in df['Nation']:
        if item not in nationlist:
            nationlist.append(item)
    #print nationlist
    #print df[(df['Age'] > 30) & (df['Nch'] > 3)][['Cost']].values
    #print df[(df['Nation']=='DE')][['Cost']].values
    avglist=[]
    for nation in nationlist:
        avg=0
        for value in df[(df['Nation']==nation)][['Cost']].values:
            avg+=value
        avg = avg/(len(df[(df['Nation']=='DE')][['Cost']].values))
        avglist.append(int(avg))
    #print avglist
    ind = np.arange(len(nationlist))
    plt.bar(ind,avglist)
    plt.ylabel('Average Cost')
    plt.xlabel('Nation')
    plt.title('Question 2: Average Cost per Nation')
    plt.xticks(ind+.5, nationlist)
    plt.savefig('question2.png')
    plt.show()

question2('cernstaff_mod.csv')

def question3(finpath='cernstaff_mod.csv'):
    '''  Small example how to read the file into a Pandas DataFrame and do some basic things.
         The same can be done directly with numpy instead of Pandas. '''
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # Learn about Pandas DataFrames and how to read in tables from ascii files
    df = pd.read_csv(finpath, index_col=False, skiprows=1,
                     names=['Cat', 'Flag', 'Age', 'Srvc', 'Nch', 'Grade', 'Step', 'Hrweek', 'Cost', 'Div', 'Nation',
                            'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'], lineterminator='\n')
    # Let's merge all 10 columns of children ages into a new DataFrame column, indeed a list:
    df['ChildrenAges'] = list(zip(df.c0, df.c1, df.c2, df.c3, df.c4, df.c5, df.c6, df.c7, df.c8, df.c9))
    gradelist=[]
    for item in df['Grade']:
        if item not in gradelist:
            gradelist.append(item)
    gradelist.sort()
    #print gradelist, len(gradelist)
    nationlist=[]
    for item in df['Nation']:
        if item not in nationlist:
            nationlist.append(item)
    #print nationlist
    shortnationlist=['DE','CH','NL']
    degradelist=[]
    chgradelist=[]
    nlgradelist=[]
    for grade in gradelist:
        degradelist.append(len(df[(df['Nation']=='DE') & (df['Grade']==grade)]))
        chgradelist.append(len(df[(df['Nation'] == 'CH') & (df['Grade'] == grade)]))
        nlgradelist.append(len(df[(df['Nation'] == 'NL') & (df['Grade'] == grade)]))
    #print degradelist
    #print chgradelist
    #print nlgradelist

    ind = np.arange(len(gradelist))
    p1 = plt.bar(ind,degradelist,color='red')
    p2 = plt.bar(ind, chgradelist,color='blue')
    p3 = plt.bar(ind, nlgradelist,color='green')
    plt.ylabel('Number of People')
    plt.xlabel('Grade')
    plt.xticks(ind + .5, gradelist)
    plt.legend((p1[0], p2[0],p3[0]), ('Nation = DE', 'Nation = CH','Natoin = NL'))
    plt.title('Question 3: Number of People with Certain Grades in Different Nations')
    plt.savefig('question3.png')
    plt.show()

question3('cernstaff_mod.csv')

def question4(finpath='cernstaff_mod.csv'):
    '''  Small example how to read the file into a Pandas DataFrame and do some basic things.
         The same can be done directly with numpy instead of Pandas. '''
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # Learn about Pandas DataFrames and how to read in tables from ascii files
    df = pd.read_csv(finpath, index_col=False, skiprows=1,
                     names=['Cat', 'Flag', 'Age', 'Srvc', 'Nch', 'Grade', 'Step', 'Hrweek', 'Cost', 'Div', 'Nation',
                            'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'], lineterminator='\n')
    agelist = []
    for item in df['Age']:
        if item not in agelist:
            agelist.append(item)
    agelist.sort()
    #print agelist, len(agelist)
    binlist = np.arange(20,70,5)
    #print binlist
    numlist=[]
    agelist=[]
    for a in range(20,65,5):
        #print a,a+5
        agelist.append(str(a)+' - '+str(a+5))
        i=0
        for age in df['Age']:
            if age>a and age<a+5:
                i+=1
        numlist.append(i)
    #print numlist
    ind = np.arange(len(numlist))
    plt.bar(ind,numlist)
    plt.ylabel('Number')
    plt.xlabel('Age')
    plt.title('Question 4 Age Distribution')
    plt.xticks(ind + .5, agelist)
    plt.savefig('question4.png')
    plt.show()

question4('cernstaff_mod.csv')

def question5(finpath='cernstaff_mod.csv'):
    '''  Small example how to read the file into a Pandas DataFrame and do some basic things.
         The same can be done directly with numpy instead of Pandas. '''
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # Learn about Pandas DataFrames and how to read in tables from ascii files
    df = pd.read_csv(finpath, index_col=False, skiprows=1,
                     names=['Cat', 'Flag', 'Age', 'Srvc', 'Nch', 'Grade', 'Step', 'Hrweek', 'Cost', 'Div', 'Nation',
                            'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'], lineterminator='\n')
    print "Question 5, number of British employees with 2 children is: " , len(df[(df['Nation'] == 'GB') & (df['Nch'] == 2)])

question5('cernstaff_mod.csv')

def question6(finpath='cernstaff_mod.csv'):
    '''  Small example how to read the file into a Pandas DataFrame and do some basic things.
         The same can be done directly with numpy instead of Pandas. '''
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # Learn about Pandas DataFrames and how to read in tables from ascii files
    df = pd.read_csv(finpath, index_col=False, skiprows=1,
                     names=['Cat', 'Flag', 'Age', 'Srvc', 'Nch', 'Grade', 'Step', 'Hrweek', 'Cost', 'Div', 'Nation',
                            'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'], lineterminator='\n')
    #print len(df[(df["Age"]>50)][['Cost']].values)
    tot=0
    for cost in df[(df["Age"]>50)][['Cost']].values:
        tot+=int(cost)
    #print tot
    avgover50=tot/len(df[(df["Age"]>50)][['Cost']].values)
    print "Question 6: Average cost of people over 50 is", avgover50
    tot1 = 0
    for cost in df[(df["Age"] < 35)][['Cost']].values:
        tot1 += int(cost)
    avgunder35= tot1/len(df[(df["Age"] < 35)][['Cost']].values)
    print "Question 6: Average cost of people under 35 is", avgunder35
question6('cernstaff_mod.csv')

def question7(finpath='cernstaff_mod.csv'):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # Learn about Pandas DataFrames and how to read in tables from ascii files
    df = pd.read_csv(finpath, index_col=False, skiprows=1,
                     names=['Cat', 'Flag', 'Age', 'Srvc', 'Nch', 'Grade', 'Step', 'Hrweek', 'Cost', 'Div', 'Nation',
                            'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'], lineterminator='\n')
    # Let's merge all 10 columns of children ages into a new DataFrame column, indeed a list:
    df['ChildrenAges'] = list(zip(df.c0, df.c1, df.c2, df.c3, df.c4, df.c5, df.c6, df.c7, df.c8, df.c9))
    agelist=[]
    for i in range(10):
        string = 'c'+str(i)
        for age in df[string]:
            agelist.append(age)
    agelist.sort()
    newagelist=[]
    for age in agelist:
        if age<100:
            newagelist.append(age)
    newagelist.sort()
    #print agelist
    #print newagelist

    numlist = []
    agelist1 = []
    for a in range(0, 45, 5):
        #print a, a + 5
        agelist1.append(str(a) + ' - ' + str(a + 5))
        i = 0
        for age in newagelist:
            if age > a and age < a + 5:
                i += 1
        numlist.append(i)
    #print numlist
    #print agelist1
    ind = np.arange(len(numlist))
    plt.bar(ind, numlist)
    plt.ylabel('Number')
    plt.xlabel('Age')
    plt.title('Question 7: Child Age Distribution')
    plt.xticks(ind + .5, agelist1)
    plt.savefig('question7.png')
    plt.show()
question7('cernstaff_mod.csv')

def question8(finpath='cernstaff_mod.csv'):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # Learn about Pandas DataFrames and how to read in tables from ascii files
    df = pd.read_csv(finpath, index_col=False, skiprows=1,
                     names=['Cat', 'Flag', 'Age', 'Srvc', 'Nch', 'Grade', 'Step', 'Hrweek', 'Cost', 'Div', 'Nation',
                            'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'], lineterminator='\n')
    agelist1=[]
    for value in df[(df['Cost']>10000)][['c0']].values:
        if value<100:
            agelist1.append(float(value))
    agelist1.sort()
    #print len(df['Cost'])
    #print len(df[(df['Cost']>10000)][['c0']].values)
    #print agelist1
    ind=np.arange(len(agelist1))
    numlist = []
    agelist = []
    for a in range(0, 50, 5):
        #print a, a + 5
        agelist.append(str(a) + ' - ' + str(a + 5))
        i = 0
        for age in agelist1:
            if age > a and age < a + 5:
                i += 1
        numlist.append(i)
    #print numlist
    ind = np.arange(len(numlist))
    plt.bar(ind, numlist)
    plt.ylabel('Number')
    plt.xlabel('Age')
    plt.title('Question 8: Ages of 2nd child of employees costing more than 10000')
    plt.xticks(ind + .5, agelist)
    plt.savefig('question8.png')
    plt.show()


question8('cernstaff_mod.csv')

def question9(finpath='cernstaff_mod.csv'):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # Learn about Pandas DataFrames and how to read in tables from ascii files
    df = pd.read_csv(finpath, index_col=False, skiprows=1,
                     names=['Cat', 'Flag', 'Age', 'Srvc', 'Nch', 'Grade', 'Step', 'Hrweek', 'Cost', 'Div', 'Nation',
                            'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'], lineterminator='\n')
    nationlist = []
    for item in df['Nation']:
        if item not in nationlist:
            nationlist.append(item)
    #print nationlist

    nationTograde={}
    for nation in nationlist:
        tot=0
        i=0
        for grade in df[(df['Nation']==nation)][['Grade']].values:
            tot+=int(grade)
            i+=1
        avg=tot/i
        nationTograde[nation]=avg
    max=0
    for key in nationTograde:
        if nationTograde[key]>max:
            bestgrade=key
            max=nationTograde[key]
    #print nationTograde
    print "Question 9: The nation with the best average grade is", bestgrade

    nationTonch = {}
    for nation in nationlist:
        tot = 0
        i = 0
        for nch in df[(df['Nation'] == nation)][['Nch']].values:
            #print nch
            tot += int(nch)
            #print tot
            i += 1
        #print tot
        #print i
        avg = tot / float(i)
        #print avg
        nationTonch[nation] = avg
    max = 0
    for key in nationTonch:
        if nationTonch[key] > max:
            mostchildren = key
            max = nationTonch[key]
    #print nationTonch
    print "Question 9: The nation with the most children is", mostchildren
    #print nationTograde


question9('cernstaff_mod.csv')

def question10(finpath='cernstaff_mod.csv'):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # Learn about Pandas DataFrames and how to read in tables from ascii files
    df = pd.read_csv(finpath, index_col=False, skiprows=1,
                     names=['Cat', 'Flag', 'Age', 'Srvc', 'Nch', 'Grade', 'Step', 'Hrweek', 'Cost', 'Div', 'Nation',
                            'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'], lineterminator='\n')
    agelist = []
    for item in df['Age']:
        if item not in agelist:
            agelist.append(item)
    agelist.sort()
    #print agelist, len(agelist)
    numlist=[]
    binlist=[]
    for a in range(0,65,5):
        #print a,a+5
        binlist.append(str(a)+' - '+str(a+5))
        i=0
        for age in df['Age']:
            if age>a and age<a+5:
                i+=1
        numlist.append(i)
    #print numlist
    #print binlist

    childagelist=[]
    for i in range(10):
        string = 'c'+str(i)
        for age in df[string]:
            childagelist.append(age)
        childagelist.sort()
        childnewagelist=[]
    for age in childagelist:
        if age<100:
            childnewagelist.append(age)
        childnewagelist.sort()
    #print childagelist
    #print childnewagelist

    childnumlist = []
    childagelist1 = []
    for a in range(0, 65, 5):
        #print a, a + 5
        childagelist1.append(str(a) + ' - ' + str(a + 5))
        i = 0
        for age in childnewagelist:
            if age > a and age < a + 5:
                i += 1
        childnumlist.append(i)
    #print childnumlist
    #print childagelist1

    ind=np.arange(len(childnumlist))
    #print ind

    p1=plt.bar(ind,numlist)
    p2=plt.bar(ind,childnumlist,color='red')
    plt.ylabel('Number')
    plt.xlabel('Age')
    plt.title('Question 10 Age Distribution')
    plt.xticks(ind + .5, agelist)
    plt.legend((p1[0], p2[0]), ('Adult Ages', 'Child Ages'))
    plt.savefig('question10.png')
    plt.show()

question10('cernstaff_mod.csv')

def question11(finpath='cernstaff_mod.csv'):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # Learn about Pandas DataFrames and how to read in tables from ascii files
    df = pd.read_csv(finpath, index_col=False, skiprows=1,
                     names=['Cat', 'Flag', 'Age', 'Srvc', 'Nch', 'Grade', 'Step', 'Hrweek', 'Cost', 'Div', 'Nation',
                            'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'], lineterminator='\n')

    agelist=[]
    for age in df['Age']:
        if age not in agelist:
            agelist.append(age)
    for i in range(10):
        string = 'c'+str(i)
        for age in df[string]:

            if float(2)*float(age) not in agelist:
                agelist.append(float(2)*float(age))
    newagelist=[]
    for age in agelist:
        if age<100:
            newagelist.append(age)
    newagelist.sort()
    #print newagelist

    numlist = []
    binlist = []
    for a in range(0, 65, 5):
        #print a, a + 5
        binlist.append(str(a) + ' - ' + str(a + 5))
        i = 0
        for age in newagelist:
            if age > a and age < a + 5:
                i += 1
        numlist.append(i)

    #print binlist, len(binlist)
    #print numlist,len(numlist)
    ind=np.arange(len(numlist))
    plt.bar(ind,numlist)
    plt.ylabel('Number')
    plt.xlabel('Age')
    plt.title('Question 11: Age Distribution of adults and childrenX2')
    plt.xticks(ind + .5, binlist)
    plt.savefig('question11.png')
    plt.show()

question11('cernstaff_mod.csv')