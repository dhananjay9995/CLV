import pandas as pd 
import matplotlib.pyplot as plt
import plotnine as pn
from plotnine import *
from datetime import datetime, timedelta
import seaborn as sns
from scipy.stats import zscore
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotnine as pn


##############################################################################################################
df = pd.read_csv("data/CDNOW_master.txt",sep="\s+",names=['customer_id', 'date', 'quantity', 'price'])

df.head()
df.info()
df.isnull().sum()

# Convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

# Calculate total spending per customer
df['total_spending'] = df['quantity'] * df['price']
##############################################################################################################
# Group by customer_id and aggregate statistics
customer_stats = df.groupby('customer_id').agg({
    'date': ['min', 'max', 'count'],  # First purchase date, last purchase date, and total number of purchases
    'quantity': 'sum',  # Total quantity purchased
    'total_spending': 'sum'  # Total spending
}).reset_index()


df.groupby('customer_id')['quantity'].sum()

customer_stats.columns = customer_stats.columns.droplevel()

customer_stats.columns = ['customer_id', 'first_purchase_date', 'last_purchase_date', 'total_purchases', 'total_quantity', 'total_spending']
customer_stats.sort_values(by='first_purchase_date',ascending=False)

# Calculate additional metrics such as average spending per purchase and average quantity per purchase
customer_stats['avg_spending_per_purchase'] = customer_stats['total_spending'] / customer_stats['total_purchases']

#customer_stats['avg_quantity_per_purchase'] = customer_stats['quantity'] / customer_stats[('date', 'count')]

# Display the resulting DataFrame
print(customer_stats.head())

df[df.duplicated()]

sns.boxplot(df['total_spending'])
plt.show()

df.sort_values(by='date',ascending=False)

index_to_drop = df[df['total_spending'] ==  127314.99].index[0]
df = df.drop(index=index_to_drop)
df.sort_values(by='date',ascending=False)

##############################################################################################################
#Monthly Spend Analysis

monthly_spend_df = df.copy()

monthly_spend_df['Month'] = df['date'].dt.strftime('%b')

monthly_spending = monthly_spend_df.groupby('Month')['total_spending'].sum().reset_index()

(
    ggplot(monthly_spending)+
    aes(x='reorder(Month, total_spending)',y='total_spending')+
    geom_bar(stat='identity')+
    xlab('Month')
)
##############################################################################################################
#RFM

max_date = df['date'].max() + pd.DateOffset(days=1) 

rfm_table = df.groupby('customer_id').agg({
    'date': lambda x: (max_date - x.max()).days,  # Recency
    'customer_id': 'count',                            # Frequency
    'total_spending': 'sum'                            # Monetary Value
})

rfm_table.rename(columns={
    'date': 'recency(in days)',
    'customer_id': 'frequency',
    'total_spending': 'monetary_value'
}, inplace=True)

rfm_table1 = rfm_table.reset_index()


sns.heatmap(rfm_table.corr())
plt.show()  #frequncy and monetary value have high correlation

##############################################################################################################
quartiles = rfm_table.quantile(q=[0.25, 0.5, 0.75])
def rfm_segment(row):
    if row['recency(in days)'] <= quartiles['recency(in days)'][0.25]:
        r_score = 4
    elif row['recency(in days)'] <= quartiles['recency(in days)'][0.50]:
        r_score = 3
    elif row['recency(in days)'] <= quartiles['recency(in days)'][0.75]:
        r_score = 2
    else:
        r_score = 1

    if row['frequency'] <= quartiles['frequency'][0.25]:
        f_score = 1
    elif row['frequency'] <= quartiles['frequency'][0.50]:
        f_score = 2
    elif row['frequency'] <= quartiles['frequency'][0.75]:
        f_score = 3
    else:
        f_score = 4

    if row['monetary_value'] <= quartiles['monetary_value'][0.25]:
        m_score = 1
    elif row['monetary_value'] <= quartiles['monetary_value'][0.50]:
        m_score = 2
    elif row['monetary_value'] <= quartiles['monetary_value'][0.75]:
        m_score = 3
    else:
        m_score = 4

    return str(r_score) + str(f_score) + str(m_score)

rfm_table1['RFM_Segment'] = rfm_table1.apply(rfm_segment, axis=1)

#Catogarize the customers
#Best Customers - meaning that they transacted recently, do so often and spend more than other customers.444, 443,434,334,344,343,433.
#High-spending New Customers – This group consists of those customers in 4-1-4 and 4-1-3. These are customers who transacted only once, but very recently and they spent a lot. 
#Lowest-Spending Active Loyal Customers – This group consists of those customers in segments 4-4-2 and 4-4-1 (they transacted recently and do so often, but spend the least). 
#Churned Best Customers – This segment consists of those customers in groups 1-4-4, 1-4-3, 1-3-4 and 1-3-3 (they transacted frequently and spent a lot, but it’s been a long time since they’ve transacted). 
#Least important Customers, the customers who dont continue after their first purchase and discontinued - 111,112,113,114,121,212

segment_mapping = {
    '444': 'Best Customers',
    '443': 'Best Customers',
    '434': 'Best Customers',
    '334': 'Best Customers',
    '343': 'Best Customers',
    '433': 'Best Customers',
    '414': 'High-spending New Customers',
    '413': 'High-spending New Customers',
    '442': 'Lowest-Spending Active Loyal Customers',
    '441': 'Lowest-Spending Active Loyal Customers',
    '144': 'Churned Best Customers',
    '143': 'Churned Best Customers',
    '134': 'Churned Best Customers',
    '144': 'Churned Best Customers',
    'default': 'Others'
}
rfm_table1['Cust_Category'] = rfm_table1['RFM_Segment'].map(segment_mapping).fillna('Others')

rfm_table1['Cust_Category'].value_counts()


segment_analysis = rfm_table1.groupby('RFM_Segment').agg({
    'recency(in days)': 'mean',
    'frequency': 'mean',
    'monetary_value': 'mean',
    'RFM_Segment': 'count'
}).rename(columns={'RFM_Segment': 'count'}).reset_index()

# Print segment analysis
print(segment_analysis)

segment_analysis['count'].sum()

##############################################################################################################
#CLV Calculation

age_table = df.groupby('customer_id').agg({
    'date': lambda x: (max_date - x.min()).days    #age
})
age_table.rename(columns={
    'date': 'age'
}, inplace=True)

#age_table.reset_index(inplace=True)   resetting index is throwing an error for merge operation

#customer_dates = df.groupby('customer_id')['date'].agg(['min', 'max'])
#customer_dates['age'] = (customer_dates['max'] - customer_dates['min']).dt.days+1
rfm_table1 = pd.merge(rfm_table1, age_table[['age']], left_on='customer_id', right_index=True, how='left')


rfm_table1['avg_purchase_value'] = rfm_table1['monetary_value'] / rfm_table1['frequency']
customer_lifespan = rfm_table1['recency(in days)'].max() - rfm_table1['recency(in days)'].min()
rfm_table1['CLV'] = rfm_table1['avg_purchase_value'] * rfm_table1['frequency'] * customer_lifespan

def segment_clv(clv):
    if clv >= rfm_table1['CLV'].quantile(0.75):
        return 'High CLV'
    elif clv >= rfm_table1['CLV'].quantile(0.50):
        return 'Medium CLV'
    elif clv >= rfm_table1['CLV'].quantile(0.25):
        return 'Low CLV'
    else:
        return 'Very Low CLV'

rfm_table1['CLV_Segment'] = rfm_table1['CLV'].apply(segment_clv)


rfm_table1.drop('predicted_clv1',axis=1,inplace=True)

##############################################################################################################
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_frequency_recency_matrix

rfm_table1 = rfm_table1[rfm_table1['monetary_value'] > 0]

rfm_table1['recency(in days)']= rfm_table1['recency(in days)']*-1
# Fit BG/NBD model
bgf = BetaGeoFitter(penalizer_coef=0.01)
bgf.fit(rfm_table1['frequency'], rfm_table1['recency(in days)'], rfm_table1['age'])

print(bgf.summary)
plot_frequency_recency_matrix(bgf)
plt.show()


# Fit Gamma-Gamma model
ggf = GammaGammaFitter(penalizer_coef=0.0)
ggf.fit(rfm_table1['frequency'], rfm_table1['monetary_value'])

# Compute CLV for next 120 days
predicted_clv = ggf.customer_lifetime_value(
    bgf, rfm_table1['frequency'], rfm_table1['recency(in days)'], rfm_table1['age'],
    rfm_table1['monetary_value'], time=120, discount_rate=0.01
)

print(predicted_clv.head())

rfm_table1['predicted_clv'] = predicted_clv

sns.heatmap(rfm_table1[['recency(in days)','frequency','monetary_value','predicted_clv']].corr(),annot=True)
plt.show()              #predicted clv is highly dependent on customers monetory value
##############################################################################################################
#Model Building to predict 

days =90
max_date = df['date'].max()
cutoff= df['date'].max()-pd.to_timedelta(days,unit="d")

#temporal split
temporal_in_df = df[df['date'] <= cutoff]
temporal_out_df  = df[df['date'] > cutoff] #dataset of customer transaction within the last 90 days

temporal_out_df.head(10)

target_df = temporal_out_df.drop('quantity',axis=1).groupby('customer_id')['price'].sum().rename({'price':'spend_90_total'}).to_frame()
target_df['spend_90_flag'] = 1
target_df.info()

#recency
max_date = temporal_in_df['date'].max()

recency_df = temporal_in_df[['customer_id','date']].groupby('customer_id').apply(lambda x:(x['date'].max()-max_date)/pd.to_timedelta(1,"day")).to_frame()\
    .set_axis(['recency'],axis=1)


frequency_df = temporal_in_df[['customer_id','date']].groupby('customer_id').count().set_axis(['frequency'],axis=1)

monetary_df=temporal_in_df[['customer_id','price']].groupby('customer_id').agg({'price': ['sum', 'mean']}).set_axis(['sum_price','mean_price'],axis=1)

features_df = pd.concat([recency_df,frequency_df,monetary_df],axis=1).merge(target_df, left_index=True, right_index =True, how="left").fillna(0)

#visulazie top 20 customers spend 
top_20 = features_df.head(20)
top_20.reset_index(inplace=True)

p = (ggplot(top_20)
     + aes(x='customer_id', y='sum_price')
     + geom_bar(stat='identity')
     )

print(p)
##############################################################################################################
# prediction model
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import GridSearchCV

X= features_df[['recency','frequency','sum_price','mean_price']]
y= features_df[['price']]

#next 90 day spend prediction
reg_xgb = XGBRegressor(
    objective='reg:squarederror',
    randomstate =123 # Loss function for regression
)

param_grid = {
    'learning_rate': [0.01,  0.1,  0.3,  0.5],
    'n_estimators': [ 300, 400,500]
}

reg_model = GridSearchCV(
    estimator=reg_xgb,
    param_grid=param_grid,
    scoring = "neg_mean_absolute_error",
    refit=True,
    cv=5
)

reg_model.fit(X, y)

reg_model.best_score_
reg_model.best_params_

predicted_90_days_price = reg_model.predict(X)

df_predicted_100_days_price = pd.DataFrame(predicted_100_days_price,columns=['predicted_100_days_price'])


features_df['predicted_90_days_price'] = predicted_90_days_price.tolist()

#PREDICTION OF whether they will spend in next 90 days

y_class= features_df['spend_90_flag']


cla_xgb = XGBClassifier(
    objective='binary:logistic',
    random_state =123
)

param_grid = {
    'learning_rate': [0.01,  0.1,  0.3,  0.5],
    'n_estimators': [ 300, 400,500]
}

cla_model = GridSearchCV(
    estimator=cla_xgb,
    param_grid=param_grid,
    scoring = "roc_auc",
    refit=True,
    cv=5
)

cla_model.fit(X, y_class)

cla_model.best_score_
cla_model.best_params_
predicted_cla = cla_model.predict_proba(X)

prediction_df = pd.concat(
    [
        pd.DataFrame(predicted_cla)[[1]].set_axis(['spend_prob'],axis=1),features_df.reset_index()
    ],axis=1
)

prediction_df.head(40)



##################################

