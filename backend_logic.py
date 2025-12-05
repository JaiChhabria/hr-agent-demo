import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import tool

# Load API Key from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ---------------------------------------------------------
# 1. HELPER: DYNAMIC DATA LOADING & CREATION
# ---------------------------------------------------------
def load_data():
    """
    Loads fresh data from CSVs every time it is called.
    Auto-creates dummy data files if they don't exist.
    """
    try:
        # --- A. DEFINE THE COMPLEX POLICY TEXT ---
        # We write this to file every time to ensure the Demo has the right rules.
        complex_policy_text = """
        HR POLICY HANDBOOK 2025 (CONFIDENTIAL)

        1. GLOBAL MOBILITY & RELOCATION
        - Domestic Transfer: Employees transferring between local offices are eligible for a $5,000 flat relocation bonus.
        - International Relocation: We support international moves ONLY if the employee switches to a "Fixed-Term Contractor" agreement.
        - Visa Support: The company DOES NOT sponsor new visas for voluntary relocation requests.
        - Cost of Living: Salary will be adjusted to the local market rate of the destination country.

        2. REIMBURSEMENT & EXPENSES
        - Client Entertainment: Cap is $100/person. Receipt required.
        - Alcohol Policy: Expenses for alcohol are STRICTLY NOT REIMBURSABLE. Any alcohol charges on receipts will be deducted.
        - Moving Expenses: For International moves, we reimburse shipping costs up to $2,000 (Receipts required).
        - Travel Per Diem: $50/day for meals during business travel.
        - Learning Budget: $1,000/year for certifications. Manager approval required.

        3. LEAVE POLICY
        - Sick Leave: 10 days/year. No notice required.
        - Casual Leave: 12 days/year. 2 days notice required.
        - Maternity: 26 weeks paid.
        
        4. CODE OF CONDUCT
        - Data Privacy: Sharing salary data is strictly prohibited.
        """
        
        with open("hr_policy.txt", "w") as f:
            f.write(complex_policy_text)

        # --- B. ENSURE BASE FILES EXIST ---
        if not os.path.exists("benefits_log.csv"):
            pd.DataFrame(columns=["Employee_ID", "Benefit_Type", "Status", "Timestamp"]).to_csv("benefits_log.csv", index=False)
            
        df_emp = pd.read_csv("employees.csv")
        df_cand = pd.read_csv("candidates.csv")
        df_onb = pd.read_csv("onboarding.csv")
        df_cont = pd.read_csv("emergency_contacts.csv")
        
        # --- C. GENERATE ATTRITION DATA (2024-2025 TRENDS) ---
        # This ensures we have data for "Year over Year" analysis
        if not os.path.exists("attrition.csv"):
            att_data = []
            depts = sorted(df_emp['Department'].unique().tolist())
            
            # Generate data for 2 full years (Jan 2024 to Dec 2025)
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2025, 12, 30)
            
            for i in range(500, 580): # 80 Ex-employees
                random_days = random.randrange((end_date - start_date).days)
                exit_date = start_date + timedelta(days=random_days)
                
                att_data.append({
                    "Exit_ID": i,
                    "Department": random.choice(depts),
                    "Exit_Date": exit_date.strftime("%Y-%m-%d"),
                    "Reason": random.choice(['Better Offer', 'Relocation', 'Higher Studies', 'Work-Life Balance', 'Involuntary']),
                    "Term_Type": random.choice(['Voluntary', 'Involuntary']),
                    "Tenure_Years": random.randint(1, 8),
                    "Manager_ID": random.choice([101, 104, 108, 110])
                })
            df_att = pd.DataFrame(att_data)
            df_att.to_csv("attrition.csv", index=False)
        else:
            df_att = pd.read_csv("attrition.csv")

        # --- D. GENERATE ENGAGEMENT DATA ---
        # Needed for the "Performance vs Engagement" insights
        if not os.path.exists("engagement.csv"):
            eng_data = []
            for eid in df_emp['Employee_ID']:
                eng_data.append({
                    "Employee_ID": eid,
                    "Engagement_Score": random.randint(1, 10), # 1-10 Scale
                    "Performance_Rating": random.choice([1, 2, 3, 3, 4, 4, 5]), # 1-5 Scale
                    "Last_Survey_Date": "2025-11-01"
                })
            df_eng = pd.DataFrame(eng_data)
            df_eng.to_csv("engagement.csv", index=False)
        else:
            df_eng = pd.read_csv("engagement.csv")

        # Clean Emails (ensure NaNs are None for easier processing)
        df_emp['Email'] = df_emp['Email'].replace({np.nan: None, "": None})
            
        return df_emp, df_cand, df_onb, df_cont, df_att, df_eng, complex_policy_text
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None, None, ""

# ---------------------------------------------------------
# 2. SHARED LOGIC (Simulate, Reset, Math)
# ---------------------------------------------------------

def reset_demo_data():
    """
    Resets the data to a 'Messy' state for the demo.
    It intentionally removes emails and deletes contact rows.
    """
    # Force delete generated files to trigger fresh creation on next load
    if os.path.exists("attrition.csv"): os.remove("attrition.csv")
    if os.path.exists("engagement.csv"): os.remove("engagement.csv")
    
    df_emp = pd.read_csv("employees.csv")
    df_cont = pd.read_csv("emergency_contacts.csv")
    
    # Corrupt Emails (Set 15% to Empty)
    for idx in df_emp.index:
        if pd.isna(df_emp.at[idx, 'Email']):
             # Temp fill to ensure we can delete randomly
             df_emp.at[idx, 'Email'] = f"user{idx}@company.com"
        
        if random.random() < 0.15: 
            df_emp.at[idx, 'Email'] = None

    # Corrupt Contacts (Drop 20% of rows)
    df_cont = df_cont.sample(frac=0.8)
    
    df_emp.to_csv("employees.csv", index=False)
    df_cont.to_csv("emergency_contacts.csv", index=False)
    
    return "🔄 **RESET COMPLETE:** Data has been corrupted and date ranges refreshed. Ready for demo."

def simulate_employee_updates_logic():
    """
    Simulates employees fixing their data.
    Returns a MARKDOWN TABLE of the updates for the Chat UI.
    """
    import faker
    fake = faker.Faker()
    
    df_emp = pd.read_csv("employees.csv")
    df_cont = pd.read_csv("emergency_contacts.csv")
    
    updated_email_records = []
    new_contact_records = []
    
    # 1. Fix Emails
    for index, row in df_emp.iterrows():
        if pd.isna(row['Email']) or row['Email'] == "":
            new_email = f"{str(row['Name']).split()[0].lower()}.{random.randint(100,999)}@company.com"
            df_emp.at[index, 'Email'] = new_email
            updated_email_records.append({
                "Name": row['Name'],
                "New_Email": new_email
            })
            
    df_emp.to_csv("employees.csv", index=False)
    
    # 2. Fix Contacts
    missing_ids = list(set(df_emp['Employee_ID']) - set(df_cont['Employee_ID']))
    new_contacts = []
    
    for emp_id in missing_ids:
        c_name = fake.name()
        c_rel = random.choice(["Spouse", "Parent"])
        new_contacts.append({
            "Employee_ID": emp_id,
            "Contact_Name": c_name,
            "Relation": c_rel,
            "Phone": fake.phone_number()
        })
        try:
            e_name = df_emp[df_emp['Employee_ID'] == emp_id]['Name'].values[0]
        except: e_name = "Unknown"
        new_contact_records.append({"Name": e_name, "New_Contact": f"{c_name} ({c_rel})"})
    
    if new_contacts:
        df_cont = pd.concat([df_cont, pd.DataFrame(new_contacts)], ignore_index=True)
        df_cont.to_csv("emergency_contacts.csv", index=False)
    
    # 3. Generate Report
    report = ["✅ **SYSTEM UPDATE SUCCESSFUL**\n"]
    if updated_email_records:
        report.append(pd.DataFrame(updated_email_records).to_markdown(index=False))
    if new_contact_records:
        report.append("\n**Added Emergency Contacts:**")
        report.append(pd.DataFrame(new_contact_records).head(5).to_markdown(index=False))
        if len(new_contact_records) > 5: report.append(f"...and {len(new_contact_records)-5} more.")
             
    if not updated_email_records and not new_contact_records:
        return "✅ **System Checked:** No missing data found."

    return "\n".join(report)

def calculate_hike_impact(emp_id: int, hike_percent: float) -> str:
    """
    Calculates salary impact using the Rich Format (Tenure, Band Position, Recommendations).
    """
    from datetime import datetime
    
    # We load data here to ensure we have the latest salaries
    df_emp, _, _, _, _, _, _ = load_data()
    
    record = df_emp[df_emp['Employee_ID'] == emp_id]
    
    if record.empty:
        return "❌ Error: Employee not found."

    # 1. Get Details
    current = record.iloc[0]['Salary']
    role = record.iloc[0]['Role']
    name = record.iloc[0]['Name']
    
    # 2. Tenure Calculation
    try:
        join_date = datetime.strptime(str(record.iloc[0]['Join_Date']), "%Y-%m-%d")
        tenure = round((datetime.now() - join_date).days / 365, 1)
    except:
        tenure = "N/A"

    # 3. Math
    hike_amt = current * (hike_percent / 100)
    new_sal = current + hike_amt
    
    # 4. Peer Analysis
    peers = df_emp[df_emp['Role'] == role]
    avg = peers['Salary'].mean()
    min_s = peers['Salary'].min()
    max_s = peers['Salary'].max()
    
    # 5. Band Logic (Simulated Band: Min-10% to Max+10%)
    band_min, band_max = min_s * 0.9, max_s * 1.1
    
    if band_max == band_min:
        pos = 1.0
    else:
        pos = (new_sal - band_min) / (band_max - band_min)
    
    # 6. Recommendation Logic
    diff = new_sal - avg
    comp_text = "ABOVE" if diff > 0 else "BELOW"
    
    if pos > 0.85:
        rec_text = "⚠️ **High Risk:** This hike pushes the employee to the top of the pay band. Ensure performance justifies this deviation."
    else:
        rec_text = "✅ **Safe:** This keeps the employee within a healthy retention range."

    # 7. Final Output String
    return f"""
    ### 📊 Compensation Impact Report: {name}
    **{role}** | ⏳ **Tenure:** {tenure} Years
    
    **1. Scenario Details**
    * **Current Salary:** ${current:,.0f}
    * **Proposed Hike:** {hike_percent}% (+${hike_amt:,.0f})
    * **New Salary:** ${new_sal:,.0f}
    
    **2. Peer & Market Context**
    * **Peer Average:** ${avg:,.0f} (for {role})
    * **Comparison:** This places them **${abs(diff):,.0f} {comp_text}** the peer average.
    
    **3. Pay Band Positioning**
    * **Band:** ${band_min:,.0f} - ${band_max:,.0f}
    * **Position:** {pos*100:.1f}th Percentile
    
    **AI Recommendation:**
    {rec_text}
    """

# ---------------------------------------------------------
# 3. TOOL DEFINITIONS (DOCSTRINGS REQUIRED)
# ---------------------------------------------------------

@tool
def draft_policy_email(query: str) -> str:
    """
    DRAFTS emails to employees based on HR Policy.
    Use this when the user asks to 'Draft a reply', 'Write an email', or 'Respond to request'.
    """
    _, _, _, _, _, _, policy_text = load_data()
    return f"""
    INSTRUCTIONS: You are an expert HR Business Partner. 
    Draft an email based on the request: "{query}"
    
    Strictly adhere to this POLICY CONTEXT:
    {policy_text}
    
    **MANDATORY FORMAT STRUCTURE:**
    1. **Subject:** [Clear Subject Line]
    2. **Salutation:** Dear [Name],
    3. **Policy Ruling:** State clearly if approved/rejected based on policy. Quote the rule if needed.
    4. **Closing:** Best regards, HR Team.
    
    TONE: Professional, Firm on policy, but Empathetic.
    """

@tool
def audit_data_integrity(query: str) -> str:
    """
    Checks for missing data and returns a Markdown Table of offenders.
    Use this for 'Audit', 'Data Quality', or 'Check missing info'.
    """
    df_emp, _, _, df_cont, _, _, _ = load_data()
    issues = []
    
    # Check Emails
    missing_email = df_emp[df_emp['Email'].isnull() | (df_emp['Email'] == "")]
    if not missing_email.empty:
        issues.append(f"**🔴 Found {len(missing_email)} employees with missing Emails:**")
        issues.append(missing_email[['Employee_ID', 'Name']].to_markdown(index=False))
    
    # Check Contacts
    all_ids = set(df_emp['Employee_ID'])
    contact_ids = set(df_cont['Employee_ID'])
    missing_ids = list(all_ids - contact_ids)
    
    if missing_ids:
        issues.append(f"\n**🟠 Found {len(missing_ids)} employees missing Emergency Contacts:**")
        # Just show IDs if list is long to avoid clutter, or names if manageable
        issues.append(f"(IDs: {missing_ids[:10]}... see database for full list)")

    if not issues: return "✅ **Data Audit Complete:** All records are clean."
    return "\n".join(issues)

@tool
def send_correction_emails(issue_summary: str) -> str:
    """
    Triggers the correction email campaign.
    Use this when the user says 'Send emails' or 'Fix data'.
    """
    return "📧 **ACTION:** Targeted emails have been sent to identified employees."

@tool
def verify_data_remediation(query: str) -> str:
    """
    Verifies if the data gaps have been closed.
    Use this to 'Verify success' or 'Check status'.
    """
    df_emp, _, _, df_cont, _, _, _ = load_data()
    m_email = df_emp[df_emp['Email'].isnull() | (df_emp['Email'] == "")].shape[0]
    m_cont = len(set(df_emp['Employee_ID']) - set(df_cont['Employee_ID']))
    
    if m_email == 0 and m_cont == 0: return "🎉 **SUCCESS:** All data gaps closed. 100% Compliant."
    return f"⚠️ **Status:** Waiting on {m_email} emails and {m_cont} contacts."

@tool
def analyze_compensation_adjustment(query: str) -> str:
    """
    Wrapper for hike logic. 
    Use this if the user asks about salary hikes in the Chat.
    """
    import re
    hike_match = re.search(r'(\d+)%', query)
    percent = float(hike_match.group(1)) if hike_match else 10.0
    id_match = re.search(r'\b(1\d{2})\b', query)
    if id_match: return calculate_hike_impact(int(id_match.group(1)), percent)
    return "⚠️ Please use the Compensation Modeler in the sidebar for this request."

@tool
def enroll_benefit(req: str) -> str:
    """
    Enrolls an employee in benefits.
    Use this for 'Enroll in Gym', 'Add insurance', etc.
    """
    return f"✅ Benefit '{req}' logged."

@tool
def read_policy(q: str) -> str:
    """
    Reads the HR Policy handbook.
    Use this for questions about rules, leave, or code of conduct.
    """
    _, _, _, _, _, _, text = load_data()
    return f"Context:\n{text}"

@tool
def check_onboarding_status(name: str) -> str:
    """
    Checks the onboarding status of a candidate.
    Use this for 'Status of Alex', 'Is John onboarded?'.
    """
    _, _, df_onb, _, _, _, _ = load_data()
    rec = df_onb[df_onb['Employee_Name'].str.contains(name, case=False, na=False)]
    if rec.empty: return "No record."
    return rec.to_markdown()

@tool
def send_reminders(act: str) -> str:
    """
    Sends generic email reminders.
    Use this for 'Nudge candidate', 'Send reminder'.
    """
    return f"🚀 Reminders sent for '{act}'."

# ---------------------------------------------------------
# 4. AGENT CONFIGURATION
# ---------------------------------------------------------
def get_hr_agent():
    # LOAD ALL DATA (7 ITEMS)
    df_emp, df_cand, df_onb, df_cont, df_att, df_eng, policy_text = load_data()
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, google_api_key=GOOGLE_API_KEY)
    
    # Pass ALL dataframes to the Analyst
    analytics_agent = create_pandas_dataframe_agent(
        llm, 
        [df_emp, df_cand, df_att, df_eng], 
        verbose=True, 
        allow_dangerous_code=True, 
        handle_parsing_errors=True
    )
    
    tools = [
        Tool(name="Analytics", func=analytics_agent.invoke, description="Data queries for Employees, Attrition, and Engagement."),
        Tool(name="Policy", func=read_policy, description="Policy queries"),
        Tool(name="Comp", func=analyze_compensation_adjustment, description="Salary queries"),
        Tool(name="Ben", func=enroll_benefit, description="Benefit queries"),
        Tool(name="Email_Drafter", func=draft_policy_email, description="Drafts emails."),
        audit_data_integrity, send_correction_emails, verify_data_remediation, check_onboarding_status, send_reminders
    ]
    
    return initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)