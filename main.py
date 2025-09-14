import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Smart Budget Calculator",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stButton > button { width: 100%; }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'income_data' not in st.session_state:
        st.session_state.income_data = []
    if 'expense_data' not in st.session_state:
        st.session_state.expense_data = []
    if 'savings_goals' not in st.session_state:
        st.session_state.savings_goals = []
    if 'investments' not in st.session_state:
        st.session_state.investments = []
    if 'debts' not in st.session_state:
        st.session_state.debts = []
    if 'expense_categories' not in st.session_state:
        st.session_state.expense_categories = [
            "Housing", "Utilities", "Transportation", "Groceries", 
            "Healthcare", "Entertainment", "Shopping", "Insurance", 
            "Education", "Dining Out", "Personal Care", "Other"
        ]
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard"
    if 'update_goal_index' not in st.session_state:
        st.session_state.update_goal_index = None
    if 'update_debt_forms' not in st.session_state:
        st.session_state.update_debt_forms = {}

initialize_session_state()

# Helper functions
def convert_to_monthly(amount, frequency):
    """Convert different frequencies to monthly amount"""
    conversions = {
        "Weekly": 4.33,
        "Bi-weekly": 2.17,
        "Monthly": 1,
        "Quarterly": 0.33,
        "Annually": 0.083
    }
    return amount * conversions.get(frequency, 1)

def calculate_loan_payment(principal, rate, months):
    """Calculate monthly payment for a loan using loan payment formula"""
    if rate == 0:
        return principal / months
    monthly_rate = rate / 100 / 12
    payment = principal * (monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)
    return payment

def calculate_remaining_balance(principal, rate, total_months, months_paid, monthly_payment):
    """Calculate remaining balance on a loan"""
    if rate == 0:
        return max(0, principal - (monthly_payment * months_paid))
    
    monthly_rate = rate / 100 / 12
    remaining_months = total_months - months_paid
    
    if remaining_months <= 0:
        return 0
    
    # Calculate remaining balance using amortization formula
    remaining_balance = monthly_payment * ((1 + monthly_rate)**remaining_months - 1) / (monthly_rate * (1 + monthly_rate)**remaining_months)
    return max(0, remaining_balance)

def calculate_debt_payoff_time(current_balance, monthly_payment, annual_rate):
    """Calculate time to pay off debt with given payment"""
    if monthly_payment <= 0 or current_balance <= 0:
        return float('inf')
    
    if annual_rate == 0:
        return current_balance / monthly_payment
    
    monthly_rate = annual_rate / 100 / 12
    if monthly_payment <= current_balance * monthly_rate:
        return float('inf')  # Payment too small to cover interest
    
    months = -np.log(1 - (current_balance * monthly_rate) / monthly_payment) / np.log(1 + monthly_rate)
    return months

def dashboard_page():
    st.header("📊 Financial Dashboard")
    
    # Calculate totals
    total_monthly_income = sum([convert_to_monthly(item['amount'], item['frequency']) for item in st.session_state.income_data])
    total_monthly_expenses = sum([convert_to_monthly(item['amount'], item['frequency']) for item in st.session_state.expense_data])
    net_cash_flow = total_monthly_income - total_monthly_expenses
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Monthly Income", f"₹{total_monthly_income:,.2f}")
    
    with col2:
        st.metric("Monthly Expenses", f"₹{total_monthly_expenses:,.2f}")
    
    with col3:
        st.metric("Net Cash Flow", f"₹{net_cash_flow:,.2f}", 
                 delta=f"₹{net_cash_flow:,.2f}" if net_cash_flow >= 0 else f"-₹{abs(net_cash_flow):,.2f}")
    
    with col4:
        savings_rate = (net_cash_flow / total_monthly_income * 100) if total_monthly_income > 0 else 0
        st.metric("Savings Rate", f"{savings_rate:.1f}%")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.expense_data:
            # Expense breakdown pie chart
            expense_df = pd.DataFrame(st.session_state.expense_data)
            expense_df['monthly_amount'] = expense_df.apply(
                lambda x: convert_to_monthly(x['amount'], x['frequency']), axis=1
            )
            fig = px.pie(expense_df, values='monthly_amount', names='category', title="Expense Breakdown")
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add expenses to see the breakdown chart")
    
    with col2:
        # Income vs Expenses comparison
        if total_monthly_income > 0 or total_monthly_expenses > 0:
            fig = go.Figure(data=[
                go.Bar(name='Income', x=['Monthly Total'], y=[total_monthly_income], marker_color='#2E8B57'),
                go.Bar(name='Expenses', x=['Monthly Total'], y=[total_monthly_expenses], marker_color='#DC143C')
            ])
            fig.update_layout(title="Income vs Expenses", barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add income and expenses to see the comparison")
    
    # Cash flow statement
    st.subheader("💸 Cash Flow Statement")
    cash_flow_data = {
        'Category': ['Income', 'Expenses', 'Net Flow'],
        'Amount': [total_monthly_income, -total_monthly_expenses, net_cash_flow],
        'Color': ['green' if x >= 0 else 'red' for x in [total_monthly_income, -total_monthly_expenses, net_cash_flow]]
    }
    
    fig = px.bar(cash_flow_data, x='Category', y='Amount', color='Color', 
                title="Monthly Cash Flow", color_discrete_map={'green': '#2E8B57', 'red': '#DC143C'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Alerts
    if st.session_state.alerts:
        st.subheader("🔔 Active Alerts")
        for alert in st.session_state.alerts:
            if alert['type'] == 'warning':
                st.warning(alert['message'])
            elif alert['type'] == 'info':
                st.info(alert['message'])

def income_page():
    st.header("💵 Income Management")
    
    # Add new income
    with st.expander("➕ Add New Income Source", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            income_name = st.text_input("Income Source Name", placeholder="e.g., Salary, Freelance")
        
        with col2:
            income_amount = st.number_input("Amount", min_value=0.0, step=0.01)
        
        with col3:
            income_frequency = st.selectbox("Frequency", ["Weekly", "Bi-weekly", "Monthly", "Quarterly", "Annually"])
        
        if st.button("Add Income Source"):
            if income_name and income_amount > 0:
                st.session_state.income_data.append({
                    'name': income_name,
                    'amount': income_amount,
                    'frequency': income_frequency,
                    'date_added': datetime.now().strftime('%Y-%m-%d')
                })
                st.success("Income source added successfully!")
                st.rerun()
            else:
                st.error("Please fill in all fields with valid values")
    
    # Display existing income sources
    if st.session_state.income_data:
        st.subheader("Current Income Sources")
        
        for i, income in enumerate(st.session_state.income_data):
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
            
            with col1:
                st.write(f"**{income['name']}**")
            
            with col2:
                st.write(f"₹{income['amount']:,.2f}")
            
            with col3:
                st.write(income['frequency'])
            
            with col4:
                monthly_amount = convert_to_monthly(income['amount'], income['frequency'])
                st.write(f"₹{monthly_amount:,.2f}/month")
            
            with col5:
                if st.button("🗑️", key=f"delete_income_{i}"):
                    st.session_state.income_data.pop(i)
                    st.rerun()
        
        # Total monthly income visualization
        total_monthly = sum([convert_to_monthly(item['amount'], item['frequency']) for item in st.session_state.income_data])
        st.metric("Total Monthly Income", f"₹{total_monthly:,.2f}")
        
        # Income breakdown chart
        income_df = pd.DataFrame(st.session_state.income_data)
        income_df['monthly_amount'] = income_df.apply(
            lambda x: convert_to_monthly(x['amount'], x['frequency']), axis=1
        )
        fig = px.pie(income_df, values='monthly_amount', names='name', title="Income Sources Breakdown")
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No income sources added yet. Add your first income source above!")

def expenses_page():
    st.header("💳 Expense Management")
    
    # Add new expense
    with st.expander("➕ Add New Expense", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            expense_name = st.text_input("Expense Name", placeholder="e.g., Rent, Groceries")
            expense_amount = st.number_input("Amount", min_value=0.0, step=0.01, key="expense_amount")
            expense_category = st.selectbox("Category", st.session_state.expense_categories)
        
        with col2:
            expense_frequency = st.selectbox("Frequency", ["Weekly", "Bi-weekly", "Monthly", "Quarterly", "Annually"], key="expense_freq")
            expense_type = st.selectbox("Type", ["Recurring", "One-time"])
            expense_limit = st.number_input("Monthly Limit (Optional)", min_value=0.0, step=0.01)
        
        # Custom category option
        if expense_category == "Other":
            custom_category = st.text_input("Custom Category Name")
            if custom_category:
                expense_category = custom_category
        
        if st.button("Add Expense"):
            if expense_name and expense_amount > 0:
                monthly_amount = convert_to_monthly(expense_amount, expense_frequency)
                st.session_state.expense_data.append({
                    'name': expense_name,
                    'amount': expense_amount,
                    'frequency': expense_frequency,
                    'category': expense_category,
                    'type': expense_type,
                    'limit': expense_limit,
                    'monthly_amount': monthly_amount,
                    'date_added': datetime.now().strftime('%Y-%m-%d')
                })
                
                # Check for limit alerts
                if expense_limit > 0 and monthly_amount > expense_limit * 0.8:
                    alert_msg = f"Warning: {expense_name} is approaching its limit (₹{monthly_amount:.2f}/₹{expense_limit:.2f})"
                    st.session_state.alerts.append({
                        'type': 'warning',
                        'message': alert_msg,
                        'date': datetime.now().strftime('%Y-%m-%d')
                    })
                
                st.success("Expense added successfully!")
                st.rerun()
            else:
                st.error("Please fill in all required fields")
    
    # Add custom category
    with st.expander("🏷️ Manage Categories"):
        new_category = st.text_input("Add New Category")
        if st.button("Add Category"):
            if new_category and new_category not in st.session_state.expense_categories:
                st.session_state.expense_categories.append(new_category)
                st.success(f"Category '{new_category}' added!")
                st.rerun()
    
    # Display expenses
    if st.session_state.expense_data:
        st.subheader("Current Expenses")
        
        # Category filter
        selected_category = st.selectbox("Filter by Category", ["All"] + st.session_state.expense_categories)
        
        filtered_expenses = st.session_state.expense_data
        if selected_category != "All":
            filtered_expenses = [exp for exp in st.session_state.expense_data if exp['category'] == selected_category]
        
        for i, expense in enumerate(filtered_expenses):
            col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 1])
            
            with col1:
                st.write(f"**{expense['name']}**")
                st.write(f"*{expense['category']}*")
            
            with col2:
                st.write(f"₹{expense['amount']:,.2f}")
            
            with col3:
                st.write(expense['frequency'])
            
            with col4:
                st.write(f"₹{expense['monthly_amount']:,.2f}/month")
            
            with col5:
                st.write(expense['type'])
            
            with col6:
                if st.button("🗑️", key=f"delete_expense_{i}"):
                    original_index = st.session_state.expense_data.index(expense)
                    st.session_state.expense_data.pop(original_index)
                    st.rerun()
        
        # Expense analytics
        col1, col2 = st.columns(2)
        
        with col1:
            # Category breakdown
            expense_df = pd.DataFrame(st.session_state.expense_data)
            category_totals = expense_df.groupby('category')['monthly_amount'].sum().reset_index()
            fig = px.pie(category_totals, values='monthly_amount', names='category', title="Expenses by Category")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Recurring vs One-time
            type_totals = expense_df.groupby('type')['monthly_amount'].sum().reset_index()
            fig = px.bar(type_totals, x='type', y='monthly_amount', title="Recurring vs One-time Expenses")
            st.plotly_chart(fig, use_container_width=True)

def savings_investments_page():
    st.header("💰 Savings & Investments")
    
    # Savings Goals Section
    st.subheader("🎯 Savings Goals")
    
    with st.expander("➕ Add New Savings Goal", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            goal_name = st.text_input("Goal Name", placeholder="e.g., Emergency Fund, Vacation")
            target_amount = st.number_input("Target Amount", min_value=0.0, step=0.01)
        
        with col2:
            current_amount = st.number_input("Current Amount", min_value=0.0, step=0.01)
            target_date = st.date_input("Target Date")
        
        with col3:
            monthly_contribution = st.number_input("Monthly Contribution", min_value=0.0, step=0.01)
        
        if st.button("Add Savings Goal"):
            if goal_name and target_amount > 0:
                st.session_state.savings_goals.append({
                    'name': goal_name,
                    'target_amount': target_amount,
                    'current_amount': current_amount,
                    'target_date': target_date.strftime('%Y-%m-%d'),
                    'monthly_contribution': monthly_contribution,
                    'date_created': datetime.now().strftime('%Y-%m-%d')
                })
                st.success("Savings goal added successfully!")
                st.rerun()
    
    # Display savings goals
    if st.session_state.savings_goals:
        for i, goal in enumerate(st.session_state.savings_goals):
            progress = (goal['current_amount'] / goal['target_amount']) * 100
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**{goal['name']}**")
                st.progress(progress / 100)
                st.write(f"₹{goal['current_amount']:,.2f} of ₹{goal['target_amount']:,.2f} ({progress:.1f}%)")
                
                # Calculate months to goal
                remaining = goal['target_amount'] - goal['current_amount']
                if goal['monthly_contribution'] > 0:
                    months_needed = remaining / goal['monthly_contribution']
                    st.write(f"Estimated completion: {months_needed:.1f} months")
            
            with col2:
                if st.button("Update", key=f"update_goal_{i}"):
                    st.session_state.update_goal_index = i
                    st.rerun()
                
                if st.button("🗑️", key=f"delete_goal_{i}"):
                    st.session_state.savings_goals.pop(i)
                    st.rerun()
    
    # Update goal form
    if st.session_state.update_goal_index is not None:
        goal_index = st.session_state.update_goal_index
        if goal_index < len(st.session_state.savings_goals):
            goal = st.session_state.savings_goals[goal_index]
            st.subheader(f"Update Goal: {goal['name']}")
            
            with st.form(f"update_form_{goal_index}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    updated_current = st.number_input("Current Amount", value=goal['current_amount'], min_value=0.0, step=0.01)
                    updated_target = st.number_input("Target Amount", value=goal['target_amount'], min_value=0.0, step=0.01)
                
                with col2:
                    updated_contribution = st.number_input("Monthly Contribution", value=goal['monthly_contribution'], min_value=0.0, step=0.01)
                    updated_date = st.date_input("Target Date", value=datetime.strptime(goal['target_date'], '%Y-%m-%d').date())
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.form_submit_button("Save Changes"):
                        st.session_state.savings_goals[goal_index].update({
                            'current_amount': updated_current,
                            'target_amount': updated_target,
                            'monthly_contribution': updated_contribution,
                            'target_date': updated_date.strftime('%Y-%m-%d')
                        })
                        st.session_state.update_goal_index = None
                        st.success("Goal updated successfully!")
                        st.rerun()
                
                with col2:
                    if st.form_submit_button("Cancel"):
                        st.session_state.update_goal_index = None
                        st.rerun()
    
    # Investments Section
    st.subheader("📈 Investment Tracking")
    
    with st.expander("➕ Add New Investment", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            investment_name = st.text_input("Investment Name", placeholder="e.g., S&P 500 ETF")
            investment_type = st.selectbox("Type", ["Stocks", "Bonds", "ETF", "Mutual Fund", "Crypto", "Real Estate"])
        
        with col2:
            initial_investment = st.number_input("Initial Investment", min_value=0.0, step=0.01)
            current_value = st.number_input("Current Value", min_value=0.0, step=0.01)
        
        with col3:
            expected_return = st.number_input("Expected Annual Return (%)", min_value=0.0, max_value=100.0, step=0.1)
        
        if st.button("Add Investment"):
            if investment_name and initial_investment > 0:
                st.session_state.investments.append({
                    'name': investment_name,
                    'type': investment_type,
                    'initial_investment': initial_investment,
                    'current_value': current_value,
                    'expected_return': expected_return,
                    'date_added': datetime.now().strftime('%Y-%m-%d')
                })
                st.success("Investment added successfully!")
                st.rerun()
    
    # Display investments
    if st.session_state.investments:
        st.subheader("Current Investments")
        
        total_invested = sum([inv['initial_investment'] for inv in st.session_state.investments])
        total_current = sum([inv['current_value'] for inv in st.session_state.investments])
        total_return = ((total_current - total_invested) / total_invested * 100) if total_invested > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Invested", f"₹{total_invested:,.2f}")
        
        with col2:
            st.metric("Current Value", f"₹{total_current:,.2f}")
        
        with col3:
            st.metric("Total Return", f"{total_return:.2f}%")
        
        # Investment breakdown
        investment_df = pd.DataFrame(st.session_state.investments)
        fig = px.pie(investment_df, values='current_value', names='name', title="Investment Portfolio Breakdown")
        st.plotly_chart(fig, use_container_width=True)

def debt_management_page():
    st.header("💳 Debt Management")
    
    # Add new debt with automatic calculations
    with st.expander("➕ Add New Debt", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            debt_name = st.text_input("Debt Name", placeholder="e.g., Credit Card, Student Loan")
            debt_type = st.selectbox("Type", ["Credit Card", "Personal Loan", "Car Loan", "Home Loan", "Student Loan", "Other"])
            principal = st.number_input("Original Loan Amount", min_value=0.0, step=0.01)
            interest_rate = st.number_input("Annual Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.01)
        
        with col2:
            loan_tenure_months = st.number_input("Loan Tenure (months)", min_value=1, step=1)
            loan_start_date = st.date_input("Loan Start Date", value=datetime.now() - timedelta(days=365))
            monthly_payment_made = st.number_input("Monthly Payment You've Been Making", min_value=0.0, step=0.01)
        
        # Calculate loan details automatically
        if st.button("Add Debt"):
            if debt_name and principal > 0 and loan_tenure_months > 0:
                # Calculate standard monthly payment for the loan
                standard_payment = calculate_loan_payment(principal, interest_rate, loan_tenure_months)
                
                # Calculate how many months have passed since loan start
                months_elapsed = max(0, (datetime.now().date() - loan_start_date).days // 30)
                months_paid = min(months_elapsed, loan_tenure_months)
                
                # Calculate current remaining balance
                if monthly_payment_made > 0:
                    current_balance = calculate_remaining_balance(
                        principal, interest_rate, loan_tenure_months, months_paid, monthly_payment_made
                    )
                else:
                    current_balance = principal
                
                # Calculate minimum payment (standard payment or current payment, whichever is higher)
                minimum_payment = max(standard_payment, monthly_payment_made) if monthly_payment_made > 0 else standard_payment
                
                st.session_state.debts.append({
                    'name': debt_name,
                    'type': debt_type,
                    'original_principal': principal,
                    'current_balance': current_balance,
                    'interest_rate': interest_rate,
                    'loan_tenure_months': loan_tenure_months,
                    'loan_start_date': loan_start_date.strftime('%Y-%m-%d'),
                    'months_paid': months_paid,
                    'monthly_payment_made': monthly_payment_made,
                    'standard_payment': standard_payment,
                    'minimum_payment': minimum_payment,
                    'date_added': datetime.now().strftime('%Y-%m-%d')
                })
                st.success("Debt added successfully!")
                st.rerun()
            else:
                st.error("Please fill in all required fields")
    
    # Display debts with enhanced information
    if st.session_state.debts:
        st.subheader("Current Debts")
        
        total_debt = sum([debt['current_balance'] for debt in st.session_state.debts])
        total_monthly_payments = sum([debt['monthly_payment_made'] for debt in st.session_state.debts])
        total_minimum_payments = sum([debt['minimum_payment'] for debt in st.session_state.debts])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Outstanding Debt", f"₹{total_debt:,.2f}")
        
        with col2:
            st.metric("Your Monthly Payments", f"₹{total_monthly_payments:,.2f}")
        
        with col3:
            st.metric("Minimum Required Payments", f"₹{total_minimum_payments:,.2f}")
        
        # Display each debt with detailed analysis
        for i, debt in enumerate(st.session_state.debts):
            with st.expander(f"📊 {debt['name']} - ₹{debt['current_balance']:,.2f} remaining", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Loan Details:**")
                    st.write(f"• Type: {debt['type']}")
                    st.write(f"• Original Amount: ₹{debt['original_principal']:,.2f}")
                    st.write(f"• Current Balance: ₹{debt['current_balance']:,.2f}")
                    st.write(f"• Interest Rate: {debt['interest_rate']:.2f}%")
                    st.write(f"• Loan Tenure: {debt['loan_tenure_months']} months")
                
                with col2:
                    st.write("**Payment Information:**")
                    st.write(f"• Standard Payment: ₹{debt['standard_payment']:,.2f}")
                    st.write(f"• Your Current Payment: ₹{debt['monthly_payment_made']:,.2f}")
                    st.write(f"• Minimum Required: ₹{debt['minimum_payment']:,.2f}")
                    st.write(f"• Months Paid: {debt['months_paid']}")
                    remaining_months = debt['loan_tenure_months'] - debt['months_paid']
                    st.write(f"• Months Remaining: {max(0, remaining_months)}")
                
                with col3:
                    st.write("**Payoff Analysis:**")
                    if debt['monthly_payment_made'] > 0:
                        # Calculate payoff time with current payment
                        current_payoff_months = calculate_debt_payoff_time(
                            debt['current_balance'], debt['monthly_payment_made'], debt['interest_rate']
                        )
                        
                        if current_payoff_months != float('inf'):
                            st.write(f"• Payoff Time: {current_payoff_months:.1f} months")
                            total_interest = (debt['monthly_payment_made'] * current_payoff_months) - debt['current_balance']
                            st.write(f"• Total Interest: ₹{total_interest:,.2f}")
                        else:
                            st.write("• ⚠️ Current payment too low!")
                        
                        # Show what happens with minimum payment
                        min_payoff_months = calculate_debt_payoff_time(
                            debt['current_balance'], debt['minimum_payment'], debt['interest_rate']
                        )
                        if min_payoff_months != float('inf'):
                            min_total_interest = (debt['minimum_payment'] * min_payoff_months) - debt['current_balance']
                            st.write(f"• With Min Payment: {min_payoff_months:.1f} months")
                            st.write(f"• Min Payment Interest: ₹{min_total_interest:,.2f}")
                
                # Payment strategy suggestions
                st.write("**💡 Payment Strategy Suggestions:**")
                if debt['monthly_payment_made'] < debt['minimum_payment']:
                    st.error(f"⚠️ Your current payment (₹{debt['monthly_payment_made']:,.2f}) is below the minimum required (₹{debt['minimum_payment']:,.2f})")
                elif debt['monthly_payment_made'] == debt['minimum_payment']:
                    extra_payment = debt['minimum_payment'] * 0.2  # Suggest 20% extra
                    st.info(f"💰 Consider paying an extra ₹{extra_payment:,.2f} per month to save on interest")
                else:
                    if 'min_total_interest' in locals() and current_payoff_months != float('inf'):
                        savings_vs_min = min_total_interest - ((debt['monthly_payment_made'] * current_payoff_months) - debt['current_balance'])
                        if savings_vs_min > 0:
                            st.success(f"✅ Great! You're saving ₹{savings_vs_min:,.2f} in interest with your current payment")
                
                # Update/Delete buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Update Payment", key=f"update_debt_btn_{i}"):
                        if i not in st.session_state.update_debt_forms:
                            st.session_state.update_debt_forms[i] = True
                        else:
                            st.session_state.update_debt_forms[i] = not st.session_state.update_debt_forms.get(i, False)
                        st.rerun()
                
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_debt_{i}"):
                        st.session_state.debts.pop(i)
                        # Clean up any update forms for this debt
                        if i in st.session_state.update_debt_forms:
                            del st.session_state.update_debt_forms[i]
                        st.rerun()
                
                # Update payment form
                if st.session_state.update_debt_forms.get(i, False):
                    st.write("**Update Monthly Payment:**")
                    with st.form(f"update_payment_form_{i}"):
                        new_payment = st.number_input(
                            "New Monthly Payment",
                            value=debt['monthly_payment_made'],
                            min_value=0.0,
                            step=0.01,
                            key=f"new_payment_input_{i}"
                        )
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.form_submit_button("Save Changes"):
                                st.session_state.debts[i]['monthly_payment_made'] = new_payment
                                st.session_state.update_debt_forms[i] = False
                                st.success("Payment updated successfully!")
                                st.rerun()
                        
                        with col2:
                            if st.form_submit_button("Cancel"):
                                st.session_state.update_debt_forms[i] = False
                                st.rerun()
        
        # Debt visualization
        if len(st.session_state.debts) > 1:
            st.subheader("📈 Debt Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Debt balances by type
                debt_df = pd.DataFrame(st.session_state.debts)
                fig = px.bar(debt_df, x='name', y='current_balance', color='type', 
                           title="Current Debt Balances", text='current_balance')
                fig.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Interest rates comparison
                fig = px.scatter(debt_df, x='current_balance', y='interest_rate', 
                               size='monthly_payment_made', hover_name='name',
                               title="Debt Balance vs Interest Rate",
                               labels={'current_balance': 'Current Balance (₹)', 'interest_rate': 'Interest Rate (%)'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Debt avalanche vs snowball strategy
            st.subheader("🎯 Debt Payoff Strategies")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🔥 Debt Avalanche (Highest Interest First):**")
                avalanche_order = sorted(st.session_state.debts, key=lambda x: x['interest_rate'], reverse=True)
                for idx, debt in enumerate(avalanche_order):
                    st.write(f"{idx+1}. {debt['name']} - {debt['interest_rate']:.2f}% interest")
                st.info("This strategy minimizes total interest paid")
            
            with col2:
                st.write("**❄️ Debt Snowball (Smallest Balance First):**")
                snowball_order = sorted(st.session_state.debts, key=lambda x: x['current_balance'])
                for idx, debt in enumerate(snowball_order):
                    st.write(f"{idx+1}. {debt['name']} - ₹{debt['current_balance']:,.2f}")
                st.info("This strategy provides psychological wins early")
    
    else:
        st.info("No debts added yet. Add your first debt above to start tracking!")

def get_expense_dataframe():
    """Return a DataFrame of all expenses with monthly amounts"""
    if not st.session_state.expense_data:
        return pd.DataFrame()
    df = pd.DataFrame(st.session_state.expense_data)
    df["MonthlyAmount"] = df.apply(
        lambda x: convert_to_monthly(x["amount"], x["frequency"]), axis=1
    )
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def create_pdf_report_with_charts(df, fig_pie):
    """Generate PDF report with totals, category breakdown, and pie chart"""
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, height - 50, "Expense Report & Analysis")
    p.drawString(50, height - 70, f"Generated on: {datetime.now().strftime('%Y-%m-%d')}")

    y_position = height - 120
    p.setFont("Helvetica", 12)

    total_expenses = df["MonthlyAmount"].sum()
    p.drawString(50, y_position, f"Total Monthly Expenses: ₹{total_expenses:,.2f}")
    y_position -= 30

    # Category breakdown
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, y_position, "Expenses by Category:")
    y_position -= 20
    p.setFont("Helvetica", 11)
    for _, row in df.groupby("category")["MonthlyAmount"].sum().reset_index().iterrows():
        p.drawString(60, y_position, f"- {row['category']}: ₹{row['MonthlyAmount']:,.2f}")
        y_position -= 18

    # Add pie chart to PDF
    img_buf = io.BytesIO()
    fig_pie.write_image(img_buf, format="PNG")
    img_buf.seek(0)
    img = ImageReader(img_buf)
    p.drawImage(img, 50, 200, width=500, height=300, preserveAspectRatio=True)

    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer


def report_analysis_page():
    st.header("📑 Expense Report & Analysis")

    df = get_expense_dataframe()
    if df.empty:
        st.info("No expenses added yet. Please add expenses to view the report & analysis.")
        return

    total_expenses = df["MonthlyAmount"].sum()
    st.subheader(f"Total Monthly Expenses: ₹{total_expenses:,.2f}")

    # Pie chart by category
    category_summary = df.groupby("category")["MonthlyAmount"].sum().reset_index()
    fig_pie = px.pie(
        category_summary,
        names="category",
        values="MonthlyAmount",
        title="Expenses by Category",
        hole=0.4,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Bar chart breakdown
    fig_bar = px.bar(
        category_summary,
        x="category",
        y="MonthlyAmount",
        title="Monthly Expense Breakdown",
        text_auto=True,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Trend over time (if date available)
    if "date" in df.columns:
        time_summary = df.groupby(df["date"].dt.to_period("M"))["MonthlyAmount"].sum()
        time_summary.index = time_summary.index.to_timestamp()

        fig_line = px.line(
            time_summary,
            x=time_summary.index,
            y=time_summary.values,
            markers=True,
            title="Expense Trend Over Time",
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # Export PDF
    if st.button("📥 Export Report to PDF"):
        pdf_buffer = create_pdf_report_with_charts(df, fig_pie)
        b64 = base64.b64encode(pdf_buffer.read()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="expense_report.pdf">Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)

# Main application
def main():
    st.title("💰 Smart Budget Calculator")
    st.markdown("Manage your finances with comprehensive tracking and analysis")
    
    # Enhanced navigation with consistent styling
    st.sidebar.markdown("### 🧭 Navigation")
    st.sidebar.markdown("---")
    
    pages = [
        ("📊 Dashboard", "Dashboard"),
        ("💵 Income", "Income"),
        ("💳 Expenses", "Expenses"),
        ("💰 Savings & Investments", "Savings & Investments"),
        ("🔒 Debt Management", "Debt Management"),
        ("📈 Reports & Analytics", "Reports & Analytics")
    ]
    
    # Create uniform navigation buttons
    for display_name, page_name in pages:
        is_current = page_name == st.session_state.current_page
        
        if st.sidebar.button(display_name, key=f"nav_{page_name}", use_container_width=True):
            st.session_state.current_page = page_name
            st.rerun()
        
        if is_current:
            st.sidebar.markdown(f"**→ Currently on: {display_name}**")
    
    st.sidebar.markdown("---")
    
    # Quick stats in sidebar
    if st.session_state.income_data or st.session_state.expense_data:
        st.sidebar.subheader("📊 Quick Stats")
        
        total_income = sum([convert_to_monthly(item['amount'], item['frequency']) for item in st.session_state.income_data])
        total_expenses = sum([convert_to_monthly(item['amount'], item['frequency']) for item in st.session_state.expense_data])
        net_flow = total_income - total_expenses
        
        st.sidebar.metric("Monthly Income", f"₹{total_income:,.0f}")
        st.sidebar.metric("Monthly Expenses", f"₹{total_expenses:,.0f}")
        st.sidebar.metric("Net Cash Flow", f"₹{net_flow:,.0f}")
    
    # Page routing
    if st.session_state.current_page == "Dashboard":
        dashboard_page()
    elif st.session_state.current_page == "Income":
        income_page()
    elif st.session_state.current_page == "Expenses":
        expenses_page()
    elif st.session_state.current_page == "Savings & Investments":
        savings_investments_page()
    elif st.session_state.current_page == "Debt Management":
        debt_management_page()
    elif st.session_state.current_page == "Reports & Analytics":
        report_analysis_page()

if __name__ == "__main__":
    main()
