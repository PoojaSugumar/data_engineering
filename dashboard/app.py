import streamlit as st
import pandas as pd
import os

# Load data
df = pd.read_csv("outputs/cleaned_finance_data.csv")
monthly_summary = pd.read_csv("outputs/monthly_summary.csv")
forecast_path = "outputs/forecast_data.csv"
forecast_data = pd.read_csv(forecast_path) if os.path.exists(forecast_path) else None
behavior_path = "outputs/classified_behavior.csv"
behavior_data = pd.read_csv(behavior_path) if os.path.exists(behavior_path) else None

# App layout
st.set_page_config(page_title="Smart Budget Planner", layout="wide")
st.title("Smart Budget Planner Dashboard")

# Sidebar navigation
tab = st.sidebar.radio("Select View", ["Overview", "Forecast", "Spending Behavior", "Budget Goals"])

# Overview Tab
if tab == "Overview":
    st.header("Monthly Expense Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Expense Trend")
        monthly_expense = df[df['Type'] == 'Expense'].groupby(df['Date'].str[:7])['Amount'].sum()
        st.line_chart(monthly_expense)

    with col2:
        st.subheader("Spending by Category")
        category_expense = df[df['Type'] == 'Expense'].groupby('Category')['Amount'].sum()
        st.bar_chart(category_expense)

# Forecast Tab
elif tab == "Forecast":
    st.header("Forecast vs Historical Average")
    if forecast_data is not None:
        st.dataframe(forecast_data)
        alerts = forecast_data[forecast_data['Overspending_Alert'] == True]
        st.warning(f"⚠️ {len(alerts)} months show potential overspending.")
    else:
        st.error("Forecast data not found. Please run the forecasting notebook.")

# Spending Behavior Tab 
elif tab == "Spending Behavior":
    st.header("Spending Behavior Classification")
    if behavior_data is not None:
        st.dataframe(behavior_data[['Year', 'Month', 'Monthly_Income', 'Monthly_Expense', 'Savings_Ratio', 'High_Value_Count', 'Predicted_Behavior']])

        st.subheader("Personalized Tips")
        for _, row in behavior_data.iterrows():
            behavior = row['Predicted_Behavior']
            date_label = f"{row['Month']}/{row['Year']}"
            if behavior == 'Saver':
                st.success(f"{date_label}: Great job! You're saving well.")
            elif behavior == 'Spender':
                st.info(f"{date_label}: Consider reducing discretionary expenses.")
            elif behavior == 'Balanced':
                st.success(f"{date_label}: You're maintaining a healthy balance.")
            elif behavior == 'Impulsive':
                st.warning(f"{date_label}: Watch out! High-value spending detected.")
    else:
        st.error("Spending behavior data not found. Please run the classification notebook.")

# Budget Goals Tab
elif tab == "Budget Goals":
    st.header("Goal-Based Budgeting Recommendations")

    st.subheader("Set Your Monthly Goal")
    income_input = st.number_input("Enter your monthly income (₹)", min_value=0.0, step=100.0)
    savings_goal = st.number_input("Enter your savings goal (₹)", min_value=0.0, step=100.0)

    if income_input > 0 and savings_goal > 0 and savings_goal < income_input:
        st.success(f"Your goal is to save ₹{savings_goal:.2f} out of ₹{income_input:.2f} income.")

        remaining_budget = income_input - savings_goal
        category_expense = df[df['Type'] == 'Expense'].groupby('Category')['Amount'].sum()
        category_ratio = category_expense / category_expense.sum()
        recommended_allocation = category_ratio * remaining_budget

        st.subheader("Recommended Budget Allocation")
        st.dataframe(pd.DataFrame({
            "Category": recommended_allocation.index,
            "Recommended Amount (₹)": recommended_allocation.values.round(2)
        }))

        st.bar_chart(recommended_allocation)
    elif savings_goal >= income_input:
        st.error("Savings goal cannot exceed or equal your income.")
    else:
        st.info("Please enter your income and savings goal to view recommendations.")
