import pandas as pd 
import matplotlib.pyplot as plt
import plotnine as pn
from plotnine import *
from datetime import datetime, timedelta


df = pd.read_csv("data/CDNOW_master.txt",sep="\s+",names=['customer_id', 'date', 'quantity', 'price'])

df.head()
df.info()
df.isna().sum()

# date time conversion
df=df.assign(date= lambda x: x['date'].astype(str)).assign(date= lambda x: pd.to_datetime(x['date']))

df_cust = df.sort_values(['customer_id','date']).groupby('customer_id').last().reset_index()

df['date'].min()
df['date'].max()

df_cust['month'] = df['date'].dt.month

monthly_total = df_cust.groupby('month')['price'].sum().reset_index()

monthly_total.plot(x='month', y='price', kind='line')
plt.show()


#cohort analysis of first 10 customers
ids = df['customer_id'].unique()
id_sel = ids[0:10]

cohort_cust = df[df['customer_id'].isin(id_sel)].groupby(['customer_id','date']).sum().reset_index()

#vislusise the purchase frequency
(ggplot(cohort_cust)
+aes(x='date',y='price',group='customer_id')
+geom_line()
+geom_point()
+facet_wrap('customer_id')
+pn.scale_x_date(date_labels='%Y',date_breaks='1 year')
)

days =100
df['date'].max()
cutoff= df['date'].max()-timedelta(days=days)
################################################################################
#temporal split
df_in_100 = df[df['date'] < cutoff]
df_out_100  = df[df['date'] >= cutoff] #dataset of customer transaction within the last 100 days

df_out_100.head(10)

#Feature Engineering(RFM)

df_test = df_out_100.groupby('customer_id')['price'].sum().reset_index().rename({'price':'last_100_days_price'},axis=1).assign(spent_100_flag =1)

#recency
max_date = df_in_100['date'].max()

recency_df = df_in_100[['customer_id','date']].groupby('customer_id').apply(lambda x:(x['date'].max()-max_date)/pd.to_timedelta(1,"day")).to_frame()\
    .set_axis(['recency'],axis=1)


frequency_df = df_in_100[['customer_id','date']].groupby('customer_id').count().set_axis(['frequency'],axis=1)

monetary_df=df_in_100[['customer_id','price']].groupby('customer_id').agg({'price': ['sum', 'mean']}).set_axis(['sum_price','mean_price'],axis=1)

merged_df = pd.concat([recency_df,frequency_df,monetary_df],axis=1).merge(df_test, left_index=True, right_index =True, how="left").fillna(0)

merged_df.drop('customer_id',axis=1, inplace=True)

#visulazie top 20 customers spend in last 100 days
top_20 = merged_df.head(20)
top_20.reset_index(inplace=True)

p = (ggplot(top_20)
     + aes(x='customer_id', y='last_100_days_price')
     + geom_bar(stat='identity')
     )

print(p)

# prediction model
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import GridSearchCV

X= merged_df[['recency','frequency','sum_price','mean_price']]
y= merged_df[['last_100_days_price']]


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

predicted_100_days_price = reg_model.predict(X)

df_predicted_100_days_price = pd.DataFrame(predicted_100_days_price,columns=['predicted_100_days_price'])


merged_df['predicted_100_days_price'] = predicted_100_days_price.tolist()

#PREDICTION OF wjether they will spend in next 100 days

y_class= merged_df[['spent_100_flag']]


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
    cv=6
)

cla_model.fit(X, y_class)

cla_model.best_score_
cla_model.best_params_
predicted_spent_100_flag = cla_model.predict_proba(X)

merged_df['predicted_spent_100_flag'] = predicted_spent_100_flag.tolist()

spend_comp=merged_df[['spent_100_flag','predicted_spent_100_flag']]

spend_comp.info()

spend_comp['spent_100_flag'] = spend_comp['spent_100_flag'].astype(int)

matches = (spend_comp['spent_100_flag'] == spend_comp['predicted_spent_100_flag']).sum()

print("Number of matching values:", matches)

21910/23570

# Assuming 'grid_search' is your fitted GridSearchCV object
# and 'X' is your input features DataFrame

# Access the feature importances from the best estimator
feature_importances = cla_model.best_estimator_.feature_importances_

# Get the feature names from the input features DataFrame
feature_names = X.columns

# Create a DataFrame mapping feature names to their importance
importance_df_cla = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

# Display the feature importances
print(importance_df_cla)


##################################
feature_importances_reg = reg_model.best_estimator_.feature_importances_

# Get the feature names from the input features DataFrame
feature_names_reg = X.columns

# Create a DataFrame mapping feature names to their importance
importance_df_reg = pd.DataFrame({
    'feature': feature_names_reg,
    'importance': feature_importances_reg
}).sort_values('importance', ascending=False)

# Display the feature importances
print(importance_df)

(
    ggplot(importance_df_reg)+
    aes(x='feature',y='importance')+
    geom_bar(stat='identity')
)
(
    ggplot(importance_df_cla)+
    aes(x='feature',y='importance')+
    geom_bar(stat='identity')
)
merged_df.drop('predicted_spent_100_flag',axis=1,inplace=True)
predictions_df = pd.concat(
    [
        pd.DataFrame(predicted_spent_100_flag)[[1]].set_axis(['predicted_spent_probability'],axis=1),
        merged_df.reset_index()
    ],axis=1
)

predictions_df.sort_values('predicted_spent_probability',ascending=False)


#people that were predicted to buy a lot, but failed to do so
predictions_df[predictions_df['last_100_days_price']==0.00].sort_values('predicted_100_days_price',ascending=False)
