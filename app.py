import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="Deriv Affiliate Dashboard", layout="wide")

# Initialize session states
if 'csv_agent' not in st.session_state:
    st.session_state.csv_agent = None
if 'selected_affiliate' not in st.session_state:
    st.session_state.selected_affiliate = None

def initialize_agent():
    """Initialize the CSV agent with Groq LLM"""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        
        os.environ["GROQ_API_KEY"] = api_key
        
        llm = ChatGroq(
            temperature=0,
            model_name="mixtral-8x7b-32768"
        )
        
        csv_agent = create_csv_agent(
            llm,
            "deriv_affiliate_data.csv",
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )
        
        return csv_agent
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None

def create_performance_chart(df, metric):
    """Create visualization based on metric type"""
    if metric == "Star Performer":
        # Get top 3 performers
        top_3_affiliates = df.groupby('affiliate_name')['commission_earned'].sum()\
            .sort_values(ascending=False).head(3).index.tolist()
        
        filtered_df = df[df['affiliate_name'].isin(top_3_affiliates)]
        
        fig = px.line(
            filtered_df.groupby(['date', 'affiliate_name'])['commission_earned'].sum().reset_index(),
            x='date', 
            y='commission_earned', 
            color='affiliate_name',
            title='Commission Trends - Top 3 Performers',
            labels={'commission_earned': 'Commission ($)', 'date': 'Date', 'affiliate_name': 'Affiliate'}
        )

    elif metric == "Cash Cow":
        # Get top 3 most consistent
        consistency = df.groupby('affiliate_name')['commission_earned'].agg(['mean', 'std']).fillna(0)
        consistency['cv'] = consistency['std'] / consistency['mean']
        top_3_consistent = consistency.sort_values('cv').head(3).index.tolist()
        
        filtered_df = df[df['affiliate_name'].isin(top_3_consistent)]
        
        fig = px.line(
            filtered_df.groupby(['date', 'affiliate_name'])['commission_earned'].sum().reset_index(),
            x='date', 
            y='commission_earned', 
            color='affiliate_name',
            title='Commission Trends - Most Consistent Performers',
            labels={'commission_earned': 'Commission ($)', 'date': 'Date', 'affiliate_name': 'Affiliate'}
        )

    elif metric == "Rising Star":
        # Calculate growth rates and get top 3 growing
        recent = df.sort_values('date').groupby('affiliate_name').tail(10)
        growth = recent.groupby('affiliate_name')['commission_earned'].mean() / \
                df.groupby('affiliate_name')['commission_earned'].mean()
        top_3_growing = growth.sort_values(ascending=False).head(3).index.tolist()
        
        filtered_df = df[df['affiliate_name'].isin(top_3_growing)]
        
        fig = px.line(
            filtered_df.groupby(['date', 'affiliate_name'])['commission_earned'].sum().reset_index(),
            x='date', 
            y='commission_earned', 
            color='affiliate_name',
            title='Commission Trends - Fastest Growing Affiliates',
            labels={'commission_earned': 'Commission ($)', 'date': 'Date', 'affiliate_name': 'Affiliate'}
        )

    elif metric == "Client Magnet":
        # Get top 3 client recruiters
        top_3_recruiters = df.groupby('affiliate_name')['new_clients'].sum()\
            .sort_values(ascending=False).head(3).index.tolist()
        
        filtered_df = df[df['affiliate_name'].isin(top_3_recruiters)]
        
        fig = px.line(
            filtered_df.groupby(['date', 'affiliate_name'])['new_clients'].sum().reset_index(),
            x='date', 
            y='new_clients', 
            color='affiliate_name',
            title='New Client Acquisition Trends - Top 3 Recruiters',
            labels={'new_clients': 'New Clients', 'date': 'Date', 'affiliate_name': 'Affiliate'}
        )

    elif metric == "Volume King":
        # Get top 3 volume generators
        top_3_volume = df.groupby('affiliate_name')['trading_volume'].sum()\
            .sort_values(ascending=False).head(3).index.tolist()
        
        filtered_df = df[df['affiliate_name'].isin(top_3_volume)]
        
        fig = px.line(
            filtered_df.groupby(['date', 'affiliate_name'])['trading_volume'].sum().reset_index(),
            x='date', 
            y='trading_volume', 
            color='affiliate_name',
            title='Trading Volume Trends - Top 3 by Volume',
            labels={'trading_volume': 'Trading Volume ($)', 'date': 'Date', 'affiliate_name': 'Affiliate'}
        )

    else:  # Market Specialist
        # Get top 3 specialists by total commission
        top_3_specialists = df.groupby('affiliate_name')['commission_earned'].sum()\
            .sort_values(ascending=False).head(3).index.tolist()
        
        filtered_df = df[df['affiliate_name'].isin(top_3_specialists)]
        
        fig = px.line(
            filtered_df.groupby(['date', 'affiliate_name'])['commission_earned'].sum().reset_index(),
            x='date', 
            y='commission_earned', 
            color='affiliate_name',
            title='Performance Trends - Top 3 Market Specialists',
            labels={'commission_earned': 'Commission ($)', 'date': 'Date', 'affiliate_name': 'Affiliate'}
        )

    # Enhance the layout for all charts
    fig.update_layout(
        xaxis_title="Date",
        legend_title="Affiliates",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified'
    )
    
    # Add gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    return fig

def analyze_data(df, metric):
    """Analyze data based on selected metric"""
    if metric == "Star Performer":
        result = df.groupby('affiliate_name')['commission_earned'].sum().sort_values(ascending=False).head(1)
        st.session_state.selected_affiliate = result.index[0]
        return f"🌟 Your star performer is {result.index[0]} with total earnings of ${result.values[0]:,.2f}"
    
    elif metric == "Cash Cow":
        consistency = df.groupby('affiliate_name')['commission_earned'].agg(['mean', 'std']).fillna(0)
        consistency['cv'] = consistency['std'] / consistency['mean']
        most_consistent = consistency.sort_values('cv').index[0]
        st.session_state.selected_affiliate = most_consistent
        return f"🐮 Your most consistent earner is {most_consistent}"
    
    elif metric == "Rising Star":
        recent = df.sort_values('date').groupby('affiliate_name').tail(10)
        growth = recent.groupby('affiliate_name')['commission_earned'].mean() / \
                df.groupby('affiliate_name')['commission_earned'].mean()
        fastest_growing = growth.sort_values(ascending=False).index[0]
        st.session_state.selected_affiliate = fastest_growing
        return f"📈 Your fastest growing affiliate is {fastest_growing}"
    
    elif metric == "Client Magnet":
        top_recruiter = df.groupby('affiliate_name')['new_clients'].sum().sort_values(ascending=False).head(1)
        st.session_state.selected_affiliate = top_recruiter.index[0]
        return f"🧲 Your top client recruiter is {top_recruiter.index[0]} with {int(top_recruiter.values[0])} new clients"
    
    elif metric == "Volume King":
        top_volume = df.groupby('affiliate_name')['trading_volume'].sum().sort_values(ascending=False).head(1)
        st.session_state.selected_affiliate = top_volume.index[0]
        return f"👑 Your highest volume generator is {top_volume.index[0]} with ${top_volume.values[0]:,.2f} in volume"
    
    elif metric == "Market Specialist":
        instrument_focus = df.groupby(['affiliate_name', 'instrument_type']).size().unstack(fill_value=0)
        specialists = instrument_focus.idxmax(axis=1)
        top_specialist = specialists.value_counts().index[0]
        st.session_state.selected_affiliate = specialists[specialists == top_specialist].index[0]
        return f"🎯 Your top market specialist focuses on {top_specialist} trading"

def get_agent_tips(metric, affiliate_name):
    """Get tips and agent suggestions for each metric"""
    tips = {
        "Star Performer": {
            "tip": f"💡 Tip: Reward {affiliate_name}'s exceptional performance!",
            "agents": [
                ("✉️ Email Agent", "Send a personalized congratulation email"),
                ("🎁 Rewards Agent", "Send special performance bonus")
            ]
        },
        "Cash Cow": {
            "tip": f"💡 Tip: Provide {affiliate_name} with advanced tools to maintain consistency!",
            "agents": [
                ("✉️ Email Agent", "Send a consistency recognition email"),
                ("🛠️ Tools Agent", "Unlock premium trading tools")
            ]
        },
        "Rising Star": {
            "tip": f"💡 Tip: Boost {affiliate_name}'s growth with extra support!",
            "agents": [
                ("✉️ Email Agent", "Send growth recognition email"),
                ("📈 Strategy Agent", "Provide personalized growth strategy")
            ]
        },
        "Client Magnet": {
            "tip": f"💡 Tip: Support {affiliate_name}'s client acquisition success!",
            "agents": [
                ("✉️ Email Agent", "Send recruitment success email"),
                ("🎯 Marketing Agent", "Provide exclusive promotional materials")
            ]
        },
        "Volume King": {
            "tip": f"💡 Tip: Enhance {affiliate_name}'s trading capabilities!",
            "agents": [
                ("✉️ Email Agent", "Send volume achievement email"),
                ("💹 Trading Agent", "Unlock advanced trading features")
            ]
        },
        "Market Specialist": {
            "tip": f"💡 Tip: Leverage {affiliate_name}'s market expertise!",
            "agents": [
                ("✉️ Email Agent", "Send expertise recognition email"),
                ("👥 Community Agent", "Feature their market insights")
            ]
        }
    }
    return tips[metric]

def main():
    st.title("📊 Deriv Affiliate Dashboard")
    
    try:
        # Load and display data
        df = pd.read_csv("deriv_affiliate_data.csv")
        
        st.subheader("Recent Activity")
        st.dataframe(df, height=200)  # Scrollable dataframe
        
        # Analysis section
        st.subheader("Quick Analysis")
        metrics = [
            "Star Performer", "Cash Cow", "Rising Star",
            "Client Magnet", "Volume King", "Market Specialist"
        ]
        
        # Create three columns for metric buttons
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        # Distribute metrics across columns
        for i, metric in enumerate(metrics):
            with cols[i % 3]:
                if st.button(metric):
                    # Show analysis
                    result = analyze_data(df, metric)
                    st.success(result)
                    
                    # Show visualization
                    fig = create_performance_chart(df, metric)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show tips and agent suggestions
                    tips = get_agent_tips(metric, st.session_state.selected_affiliate)
                    st.info(tips["tip"])
                    
                    # Display agent options
                    st.write("🤖 Available Agents:")
                    for agent_name, agent_desc in tips["agents"]:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"{agent_name}: {agent_desc}")
                        with col2:
                            st.button("Launch", key=f"{metric}_{agent_name}")
        
        # Chat interface
        st.subheader("💬 Chat with Your Data")
        custom_question = st.text_input("Ask anything about your affiliate performance:")
        if custom_question:
            if st.session_state.csv_agent:
                with st.spinner("Analyzing..."):
                    response = st.session_state.csv_agent(custom_question)
                st.success(response)
            else:
                st.error("Chat agent initialization failed. Please check your GROQ_API_KEY.")
                
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

if __name__ == "__main__":
    main()
