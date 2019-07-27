import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
import seaborn as sns
from sklearn.metrics import r2_score 
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sys import exit
import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from sklearn import preprocessing
from sklearn import metrics
import tkinter as tk 
from tkinter import *
from PIL import ImageTk, Image

import warnings
warnings.filterwarnings('ignore')


    
class mclass:
    df=pd.DataFrame()   
    def getGUI(self,clf_rf):
    
        import tkinter as tk     
        from PIL import ImageTk, Image
        # tkinter GUI
        root= tk.Tk() 
        root.title("Loan Prediction ")

        canvas1 = tk.Canvas(root, width = 700, height = 600)
        canvas1.pack()

        # loan_amnt label and input box
        label1 = tk.Label(root, text='Loan_amnt', anchor=W, width=15)
        canvas1.create_window(100, 100, window=label1)

        v1 = DoubleVar(root, value=10000)
        entry1 = tk.Entry (root,textvariable=v1) # create 1st entry box
        canvas1.create_window(270, 100, window=entry1)

        # total_rec_prncp label and input box
        label2 = tk.Label(root, text='Total_rec_prncp', anchor=W, width=15)
        canvas1.create_window(100, 120, window=label2)


        v2 = DoubleVar(root, value=10000.00)
        entry2 = tk.Entry (root,textvariable=v2) # create 2nd entry box
        canvas1.create_window(270, 120, window=entry2)

        # last_pymnt_amnt sugar label and input box
        label3 = tk.Label(root, text='Last_pymnt_amnt', anchor=W, width=15)
        canvas1.create_window(100, 140, window=label3)


        v3 = DoubleVar(root, value=370.46)
        entry3 = tk.Entry (root,textvariable=v3) # create 2nd entry box
        canvas1.create_window(270, 140, window=entry3)

        # total_pymnt_inv label and input box
        label4 = tk.Label(root, text='Total_pymnt_inv', anchor=W, width=15)
        canvas1.create_window(100, 160, window=label4)


        v4 = DoubleVar(root, value=12226.30)
        entry4 = tk.Entry (root,textvariable=v4) # create 2nd entry box
        canvas1.create_window(270, 160, window=entry4)

        # total_pymnt label and input box
        label5 = tk.Label(root, text='Total_pymnt', anchor=W, width=15)
        canvas1.create_window(100, 180, window=label5)


        v5 = DoubleVar(root, value=12226.30)
        entry5 = tk.Entry (root,textvariable=v5) # create 2nd entry box
        canvas1.create_window(270, 180, window=entry5)


        # grade_num label and input box
        label6 = tk.Label(root, text='Grade_num:', anchor=W, width=15)
        canvas1.create_window(100, 200, window=label6)

        v6 = IntVar(root, value=2)
        entry6 = tk.Entry (root,textvariable=v6) # create 2nd entry box
        canvas1.create_window(270, 200, window=entry6)


        # sub_grade_num label and input box
        label7 = tk.Label(root, text='Sub_Grade_num', anchor=W, width=15)
        canvas1.create_window(100, 220, window=label7)

        v7 = IntVar(root, value=10)
        entry7 = tk.Entry (root,textvariable=v7) # create 2nd entry box
        canvas1.create_window(270, 220, window=entry7)

        # revol_bal label and input box
        label9= tk.Label(root, text='Revol_bal', anchor=W, width=15)
        canvas1.create_window(100, 240, window=label9)

        v9 = DoubleVar(root, value=5598)
        entry9 = tk.Entry (root,textvariable=v9) # create 2nd entry box
        canvas1.create_window(270, 240, window=entry9)

        # annual_inc label and input box
        label10= tk.Label(root, text='Annual_inc', anchor=W, width=15)
        canvas1.create_window(450, 140, window=label10)


        v10 = DoubleVar(root, value=49200.0)
        entry10 = tk.Entry (root,textvariable=v10) # create 2nd entry box
        canvas1.create_window(600, 140, window=entry10)


        # funded_amnt label and input box
        label21= tk.Label(root, text='Funded_amnt', anchor=W, width=15)
        canvas1.create_window(450, 260, window=label21)

        v21 = IntVar(root, value=10000)
        entry21 = tk.Entry (root,textvariable=v21) # create 2nd entry box
        canvas1.create_window(600, 260, window=entry21)

        # total_acc label and input box
        label11= tk.Label(root, text='Total_acc', anchor=W, width=15)
        canvas1.create_window(450, 100, window=label11)

        v11 = DoubleVar(root, value=37.0)
        entry11 = tk.Entry (root,textvariable=v11) # create 2nd entry box
        canvas1.create_window(600, 100, window=entry11)

        # mths_since_last_delinq label and input box
        label12= tk.Label(root, text='Months_since_last_delinq', anchor=W, width=15)
        canvas1.create_window(450, 120, window=label12)


        v12 = DoubleVar(root, value=35.0)
        entry12 = tk.Entry (root,textvariable=v12) # create 2nd entry box
        canvas1.create_window(600, 120, window=entry12)

        # emp_length_num label and input box
        label14= tk.Label(root, text='Emp_length_num', anchor=W, width=15)
        canvas1.create_window(450, 160, window=label14)

        v14 = IntVar(root, value=11)
        entry14 = tk.Entry (root,textvariable=v14) # create 2nd entry box
        canvas1.create_window(600, 160, window=entry14)

        # pub_rec_zero label and input box
        label15= tk.Label(root, text='Pub_rec_zero', anchor=W, width=15)
        canvas1.create_window(450, 180, window=label15)

        v15 = DoubleVar(root, value=1.0)
        entry15 = tk.Entry (root,textvariable=v15) # create 2nd entry box
        canvas1.create_window(600, 180, window=entry15)


        # funded_amnt_inv label and input box
        label16= tk.Label(root, text='Funded_amnt_inv', anchor=W, width=15)
        canvas1.create_window(450, 200, window=label16)


        v16 = IntVar(root, value=10000)
        entry16 = tk.Entry (root,textvariable=v16) # create 2nd entry box
        canvas1.create_window(600, 200, window=entry16)

        # collection_recovery_fee label and input box
        label17= tk.Label(root, text='Collection_recovery_fee', anchor=W, width=15)
        canvas1.create_window(450, 220, window=label17)

        v17 = DoubleVar(root, value=0.0)
        entry17 = tk.Entry (root,textvariable=v17) # create 2nd entry box
        canvas1.create_window(600, 220, window=entry17)

        # recoveries label and input box
        label18= tk.Label(root, text='Recoveries', anchor=W, width=15)
        canvas1.create_window(450, 240, window=label18)


        v18 = DoubleVar(root, value=0.0)
        entry18 = tk.Entry (root,textvariable=v18) # create 2nd entry box
        canvas1.create_window(600, 240, window=entry18)

        label_Prediction = tk.Label(root, text= '')
        canvas1.create_window(350, 50, window=label_Prediction)


        def SetToZero():
            clear_widget_text(label_Prediction)    
            entry1.delete(0,END)
            entry2.delete(0,END)
            entry3.delete(0,END)
            entry4.delete(0,END)
            entry5.delete(0,END)
            entry6.delete(0,END)
            entry7.delete(0,END)
            entry9.delete(0,END)
            entry10.delete(0,END)
            entry21.delete(0,END)
            entry11.delete(0,END)
            entry12.delete(0,END)
            entry14.delete(0,END)
            entry15.delete(0,END)
            entry16.delete(0,END)
            entry17.delete(0,END)
            entry18.delete(0,END) 

        def ini_bad():
            SetToZero()
            entry1.insert(END,5000)
            entry2.insert(END,629.05)
            entry3.insert(END,123.65)
            entry4.insert(END,1609.12)
            entry5.insert(END,1609.12)
            entry6.insert(END,3)
            entry7.insert(END,16)
            entry9.insert(END,4345)
            entry10.insert(END,50004.0)
            entry21.insert(END,5000)
            entry11.insert(END,22.0)
            entry12.insert(END,20.0)
            entry14.insert(END,3)
            entry15.insert(END,1.0)
            entry16.insert(END,5000)
            entry17.insert(END,2.3)
            entry18.insert(END,260.96)

        def init_good():
            SetToZero()
            entry1.insert(END,10000)
            entry2.insert(END,10000.00)
            entry3.insert(END,370.46)
            entry4.insert(END,12226.30)
            entry5.insert(END,12226.30)
            entry6.insert(END,2)
            entry7.insert(END,10)
            entry9.insert(END,5598)
            entry10.insert(END,49200.0)
            entry21.insert(END,10000)
            entry11.insert(END,37.0)
            entry12.insert(END,35.0)
            entry14.insert(END,11)
            entry15.insert(END,1.0)
            entry16.insert(END,10000)
            entry17.insert(END,0.0)
            entry18.insert(END,0.0)

        def clear_widget_text(widget):
            widget['text'] = ""
            defaultbg = root.cget('bg')
            label_Prediction['bg']=defaultbg

        def open_window (): 
            root.destroy() 
            window= Tk()
            start= sclass(window,self.df)
		
        def close_main():
            root.destroy()
            os._exit(1)		


        def values(): 
            global New_loan_amnt #our 1st input variable
            New_loan_amnt= float(entry1.get()) 

            global New_total_rec_prncp #our 2nd input variable
            New_total_rec_prncp = float(entry2.get()) 

            global New_last_pymnt_amnt #our 3rd input variable
            New_last_pymnt_amnt = float(entry3.get()) 

            global New_total_pymnt_inv #our 4th input variable
            New_total_pymnt_inv = float(entry4.get()) 

            global New_total_pymnt #our 5st input variable
            New_total_pymnt = float(entry5.get()) 

            global New_grade_num #our 6 input variable
            New_grade_num = float(entry6.get()) 

            global New_sub_grade_num #our 7 input variable
            New_sub_grade_num = float(entry7.get()) 

            global New_revol_bal #our 9 input variable
            New_revol_bal = float(entry9.get()) 

            global New_annual_inc #our 10 input variable
            New_annual_inc = float(entry10.get()) 

            global New_funded_amnt #our 21 input variable
            New_funded_amnt = float(entry21.get()) 

            global New_total_acc #our 11 input variable
            New_total_acc = float(entry11.get()) 

            global New_mths_since_last_delinq #our 12 input variable
            New_mths_since_last_delinq = float(entry12.get())

            global New_emp_length_num #our 14 input variable
            New_emp_length_num = float(entry14.get())

            global New_pub_rec_zero #our 14 input variable
            New_pub_rec_zero = float(entry15.get())

            global New_funded_amnt_inv #our 16 input variable
            New_funded_amnt_inv = float(entry16.get())

            global New_collection_recovery_fee #our 17 input variable
            New_collection_recovery_fee = float(entry17.get())

            global New_recoveries #our 18 input variable
            New_recoveries = float(entry18.get())

            global to_predict

            to_predict=[[New_loan_amnt
                                    ,New_total_rec_prncp    
                                    ,New_last_pymnt_amnt   
                                    ,New_total_pymnt_inv 
                                    ,New_total_pymnt    
                                    ,New_grade_num  
                                    ,New_sub_grade_num        
                                    ,New_revol_bal               
                                    ,New_annual_inc          
                                    ,New_total_acc              
                                    ,New_mths_since_last_delinq  
                                    ,New_emp_length_num          
                                    ,New_pub_rec_zero          
                                    ,New_funded_amnt_inv
                                    ,New_collection_recovery_fee
                                    ,New_recoveries
                                    ,New_funded_amnt]]

            predicition=self.GUI_prediction(to_predict,clf_rf)  
            bgval='gray'
            if predicition ==1:
                val='Good Loan'
                bgval='green'
            if predicition ==0:
                val='Bad Loan'
                bgval='red'
            Prediction_result  = val
            label_Prediction['text']=Prediction_result
            label_Prediction['bg']=bgval
            #canvas1.create_window(350, 50, window=label_Prediction)



        button1 = tk.Button (root, text='Predict Loan',command=values, bg='orange',width = 15) # button to call the 'values' command above 
        canvas1.create_window(100, 450, window=button1)

        button2 = tk.Button (root, text='Do some Plotting',command=open_window, bg='orange',width = 15) # button to call the 'values' command above 
        canvas1.create_window(100, 500, window=button2)


        reset = tk.Button(root, text='Reset ', command=SetToZero,bg='orange' ,width = 15)
        canvas1.create_window(640, 450, window=reset) 

        init = tk.Button(root, text=' init to bad', command=ini_bad,bg='red',width = 15)
        canvas1.create_window(640, 490, window=init) 

        init = tk.Button(root, text=' init to good', command=init_good,bg='green' ,width = 15)
        canvas1.create_window(640, 520, window=init) 

        button3 = tk.Button (root, text='Close',bg='orange', command=close_main,width = 15)
        canvas1.create_window(640, 560, window=button3) 


        image = Image.open("C://Users/sbhar/Desktop/DS/Lab/Data-Science-Sydney-Curriculum/Data-Science-Sydney-Curriculum/Capstone Project/Prediction.png")
        photo = ImageTk.PhotoImage(image, master=root)
        label = tk.Label(root, image=photo)
        label.image = image

        label.pack(side = "top")
        canvas1.create_window(350, 450, window=label)
        
        root.mainloop()      
        
    def DataPreperation(self,df):
        df['issue_d']=pd.to_datetime(df['issue_d'])
        df['last_pymnt_d']=pd.to_datetime(df['last_pymnt_d'])
        df['final_d']=pd.to_datetime(df['final_d'])
        df['issue_year']=df['issue_d'].dt.year
        df['income_category'] = np.nan    
        df['income_category'][df['annual_inc'] <= 100000] = 'Low'
        df['income_category'][(df['annual_inc'] > 100000) & (df['annual_inc'] <= 200000)] = 'Medium'
        df['income_category'][ df['annual_inc'] > 200000]   = 'High'
        df['good_loan'] = np.where((df.loan_status == 'Fully Paid') |
                               (df.loan_status == 'Does not meet the credit policy. Status:Fully Paid'), 1, 0)
        from sklearn.preprocessing import LabelEncoder
        features_enc = ['grade',                     # grade of the loan
                    'sub_grade'          # sub-grade of the loan        
                   ]

        le = LabelEncoder()
        names=[]
        dff = pd.DataFrame()
        for x in features_enc:
            var=x+'_num'
            names.append(var)
            df[var]=le.fit_transform(df[x])


    def show_summary_report(self,actual, prediction):

        if isinstance(actual, pd.Series):
            actual = actual.values.astype(int)
        prediction = prediction.astype(int)

        print('Accuracy : %.4f [TP / N] Proportion of predicted labels that match the true labels. Best: 1, Worst: 0' % accuracy_score(actual, prediction))
        print('Precision: %.4f [TP / (TP + FP)] Not to label a negative sample as positive.        Best: 1, Worst: 0' % precision_score(actual, prediction))
        print('Recall   : %.4f [TP / (TP + FN)] Find all the positive samples.                     Best: 1, Worst: 0' % recall_score(actual, prediction))
        print('ROC AUC  : %.4f                                                                     Best: 1, Worst: < 0.5' % roc_auc_score(actual, prediction))
        print('-' * 107)
        print('TP: True Positives, FP: False Positives, TN: True Negatives, FN: False Negatives, N: Number of samples')

        # Confusion Matrix
        mat = confusion_matrix(actual, prediction)

        # Precision/Recall
        precision, recall, _ = precision_recall_curve(actual, prediction)
        average_precision = average_precision_score(actual, prediction)

        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(actual, prediction)
        roc_auc = auc(fpr, tpr)


        # plot
        fig, ax = plt.subplots(1, 2, figsize = (18, 6))
        fig.subplots_adjust(left = 0.02, right = 0.98, wspace = 0.2)

        # Confusion Matrix
        sns.heatmap(mat.T, square = True, annot = True, fmt = 'd', cbar = False, cmap = 'Blues', ax = ax[0])

        ax[0].set_title('Confusion Matrix')
        ax[0].set_xlabel('True label')
        ax[0].set_ylabel('Predicted label')

        # ROC
        ax[1].plot(fpr, tpr, color = 'darkorange', lw = 2, label = 'ROC curve (AUC = %0.2f)' % roc_auc)
        ax[1].plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
        ax[1].set_xlim([0.0, 1.0])
        ax[1].set_ylim([0.0, 1.0])
        ax[1].set_xlabel('False Positive Rate')
        ax[1].set_ylabel('True Positive Rate')
        ax[1].set_title('Receiver Operating Characteristic')
        ax[1].legend(loc = 'lower right')

        plt.show()



    #The Pack geometry manager packs widgets in rows or columns.
    #panel.pack(side = "bottom", fill = "both", expand = "yes")

   
    
    def importdata(self):
        filename='C://Users/sbhar/Desktop/DS/Lab/Data-Science-Sydney-Curriculum/Data-Science-Sydney-Curriculum/Capstone Project/LoanData/lending-club-data.csv'
        df = pd.read_csv(filename,low_memory=False)
        return df

    def cleanData(self,df):
        # Drop irrelevant columns
        df.drop(['id', 'member_id', 'emp_title', 'url', 'desc', 'zip_code', 'title'], axis=1, inplace=True)
        missing_col=pd.DataFrame()
        missing_col=self.missing_data(df)
        to_del=np.array(missing_col[missing_col['Percent']>80].index)
        df.drop(df[to_del], 1, inplace=True)
        df.dropna(axis=0,inplace=True)
        return df
    
    def missing_data(self,data):
        total = data.isnull().sum().sort_values(ascending = False)
        percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
        return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
  
    
    def splitdataset(self,df):
        y=df['good_loan']

        X=df[[

            'loan_amnt'
            ,'total_rec_prncp'    
            ,'last_pymnt_amnt'   
            ,'total_pymnt_inv' 
            ,'total_pymnt'    
            ,'grade_num'  
            ,'sub_grade_num'        
            ,'revol_bal'               
            ,'annual_inc'          
            ,'total_acc'              
            ,'mths_since_last_delinq'  
            ,'emp_length_num'          
            ,'pub_rec_zero'          
        ,'funded_amnt_inv'
        ,'collection_recovery_fee'
        ,'recoveries'       
        ,'funded_amnt' ]]

        scaler = preprocessing.MinMaxScaler()

        scaled_df = scaler.fit_transform(X)

        # Spliting the dataset into train and test 
        X_train, X_test, y_train, y_test = train_test_split( scaled_df, y, test_size = 0.2, random_state = 42) 

        return X_train, X_test, y_train, y_test

    
    def train_using_Logestic(self,X_train, y_train): 
        # Create a model for Linear Regression   
        Logestic = LogisticRegression() 
        # Fit the model with the Training data
        Logestic.fit(X_train,y_train)
        return Logestic       


    def train_using_gini(self,X_train, X_test, y_train): 

        # Creating the classifier object 
        clf_gini = tree.DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=5,class_weight='balanced', min_samples_leaf=5) 

        # Performing training 
        clf_gini.fit(X_train, y_train) 
        return clf_gini 

    # Function to perform training with entropy. 
    def tarin_using_entropy(self,X_train, X_test, y_train):   
        # Decision tree with entropy 
        clf_entropy = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 100, class_weight='balanced',max_depth = 5, min_samples_leaf = 5) 
         # Performing training 
        clf_entropy.fit(X_train, y_train) 
        return clf_entropy 

    def train_using_RandomForest(self,X_train,  y_train):
        rf = RandomForestClassifier(n_estimators = 20)
        rf.fit(X_train, y_train)
        return rf



    # Function to make predictions 
    def prediction(self,X_test, clf_object): 

        # Predicton on test with giniIndex 
        y_pred = clf_object.predict(X_test) 
        print("Predicted values:") 
        print(y_pred) 
        return y_pred 
        # Function to make predictions 
    def GUI_prediction(self,to_predict,clf_rf): 

        # Predicton on test with giniIndex 
        y_pred = clf_rf.predict(to_predict) 
        return y_pred 
    
    # Driver code 
    def main(self):
        p=mclass()
        df = p.importdata()
        df= p.cleanData(df)
        p.DataPreperation(df)
        X_train, X_test, y_train, y_test = p.splitdataset(df)
        clf_rf=p.train_using_RandomForest(X_train,  y_train)
        p.df=df
        p.getGUI(clf_rf)
		
class sclass:
    df=pd.DataFrame()
    figureObject, axesObject = plt.subplots()
    
    def __init__(self,window,df):   
        self.df=df
        LARGE_FONT= ("Verdana", 12)
        window.title("Pie/Box plot for Good/Bad loan.")
        self.window = window   
        
        canvas1 = tk.Canvas(window, width = 400, height = 400)
        canvas1.pack()
        
        label = Label(window, text="Data Visualisation", font=LARGE_FONT)   
        canvas1.create_window(100, 100, window=label)
        label1= Label(window, text="Select Year",  anchor=W, width=15)  
        canvas1.create_window(100, 200, window=label1)
        label2= tk.Label(window, text='Select Feature', anchor=W, width=15)
        canvas1.create_window(100, 250, window=label2)
        
        
        
        #self.box = Entry(window)
        symbol = np.sort(np.array(df['issue_year'].unique()))
        Category_list=['grade','home_ownership','is_inc_v','purpose','issue_year','income_category']
        
        var=StringVar(window)
        var.set("   ")
        
        var2=StringVar(window)
        var2.set("   ")
       
        
        drop_menu  = OptionMenu(window, var, *symbol, command=self.plotit)
        canvas1.create_window(200, 200, window=drop_menu)
        drop_menu2 = OptionMenu(window, var2, *Category_list, command=self.plotitcateg)
        canvas1.create_window(200, 250, window=drop_menu2)
        button3 = Button (window, text='Close',bg='orange', command=self.close,width = 15)
  
        
        
        button3.pack(pady=50,padx=50)
        window.mainloop()
        
        
    def plot_pie(self,year_):
        
        
        df=self.df
        self.figureObject, self.axesObject = plt.subplots(figsize=(8,6))
        label=df[df['issue_year']==year_].groupby(['good_loan'])['loan_amnt'].sum()
        labels = 'Bad Loan\n' + str(label[0])+' USD', 'Good Loan\n' +str(label[1])+' USD'
        explode = (0.1, 0)  
        X=df[df['issue_year']==year_].groupby(['good_loan'])['loan_amnt'].sum()/df[df['issue_year']==year_]['loan_amnt'].sum()
        
        self.axesObject.pie(X,explode=explode,autopct='%1.1f%%',labels=labels,shadow=True,startangle=90)
        title='Year: '+ str(year_)
        plt.title(title)
        
        
        
        newWindow = tk.Toplevel(self.window)       
        
        
        canvas = FigureCanvasTkAgg(self.figureObject, master=newWindow)
        canvas.get_tk_widget().pack()
        
        canvas.draw()
		
    def Plot_categories (self,feature):
        
        cmap = plt.cm.coolwarm_r
        df=self.df
        loans_by_grade = df.groupby([feature, 'good_loan']).size()
        #Looking the count of defaults though the issue_d that is The month which the loan was funded
        self.figureObject, self.axesObject = plt.subplots(figsize=(8,6))
        g=self.axesObject
        loans_by_grade.unstack().plot(kind='bar',colormap=cmap, ax=g, grid=False)
        g.set_xticklabels(g.get_xticklabels(),rotation=90)
        g.set_xlabel(feature, fontsize=10)
        g.set_ylabel("Count", fontsize=10)
        g.legend(loc='upper left')
        g.set_title("Analysing Loan Status by", fontsize=10)
        
        newWindowbox = tk.Toplevel(self.window)         
        
        canvas2 = FigureCanvasTkAgg(self.figureObject, master=newWindowbox)
        canvas2.get_tk_widget().pack()
        
        canvas2.draw()
        
        
        
    def plotitcateg(self,Category_list):
        self.clearPlotPage()
        self.Plot_categories(Category_list)     		
        
        
    def plotperc(self,year_):
        self.plot_pie(year_)       
   
        
           
    def plotit(self,symbol):
        self.clearPlotPage()
        self.plotperc(symbol)
    
    def clearPlotPage(self):
        self.axesObject.clear()
	
    def close(self):
        self.window.destroy()
        os._exit(1)
		
p=mclass()
p.main()
