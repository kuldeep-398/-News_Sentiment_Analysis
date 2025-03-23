"""
Simplified Streamlit application for news sentiment analysis.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from utils import (
    process_company_news,
    generate_summary_for_tts,
    generate_hindi_tts
)

# Set page config
st.set_page_config(
    page_title="News Sentiment Analysis",
    page_icon="📰",
    layout="wide"
)

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Title and description
st.title("News Sentiment Analysis")
st.write("Analyze news articles about companies, perform sentiment analysis, and generate Hindi TTS summaries.")

# Sidebar
st.sidebar.title("Settings")

# Company input
company_name = st.sidebar.text_input("Enter Company Name", "Tesla")

# Number of articles
num_articles = st.sidebar.slider("Number of Articles", min_value=5, max_value=20, value=10)

# Analysis button
if st.sidebar.button("Analyze News"):
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Update status
    status_text.text("Searching for news articles...")
    progress_bar.progress(10)
    
    try:
        # Process news articles
        result = process_company_news(company_name, num_articles)
        
        # Update progress
        status_text.text("Generating Hindi TTS summary...")
        progress_bar.progress(80)
        
        # Generate TTS summary
        summary_text = generate_summary_for_tts(result)
        audio_file = f"data/{company_name.lower()}_summary.mp3"
        
        # Generate Hindi TTS
        generate_hindi_tts(summary_text, audio_file)
        
        # Update result with audio file path
        result["audio"] = audio_file
        
        # Update progress
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # Display results
        st.header(f"Analysis Results for {result['company']}")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Articles", "Comparative Analysis", "Hindi TTS"])
        
        # Tab 1: Articles
        with tab1:
            # Display articles
            for i, article in enumerate(result["articles"]):
                with st.expander(f"{i+1}. {article['title']}", expanded=i==0):
                    st.write(f"**Source:** {article['source']}")
                    st.write(f"**Date:** {article.get('date', 'N/A')}")
                    st.write(f"**Summary:** {article['summary']}")
                    
                    # Display sentiment with color
                    sentiment = article['sentiment']
                    if sentiment == "Positive":
                        st.markdown(f"**Sentiment:** <span style='color:green'>{sentiment}</span>", unsafe_allow_html=True)
                    elif sentiment == "Negative":
                        st.markdown(f"**Sentiment:** <span style='color:red'>{sentiment}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Sentiment:** <span style='color:gray'>{sentiment}</span>", unsafe_allow_html=True)
                    
                    # Display topics
                    st.write("**Topics:**")
                    for topic in article['topics']:
                        st.markdown(f"<span style='background-color:#E3F2FD;color:#1565C0;padding:5px 10px;border-radius:15px;margin-right:5px;font-size:0.8rem;'>{topic}</span>", unsafe_allow_html=True)
                    
                    st.write(f"[Read Full Article]({article['url']})")
        
        # Tab 2: Comparative Analysis
        with tab2:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Sentiment distribution
                st.subheader("Sentiment Distribution")
                sentiment_counts = result["comparative_sentiment_score"].get("sentiment_distribution", {})
                
                if sentiment_counts:
                    # Create DataFrame
                    df = pd.DataFrame({
                        "Sentiment": list(sentiment_counts.keys()),
                        "Count": list(sentiment_counts.values())
                    })
                    
                    # Set colors
                    colors = ["#4CAF50", "#F44336", "#9E9E9E"]
                    
                    # Create figure
                    fig, ax = plt.subplots(figsize=(8, 5))
                    bars = ax.bar(df["Sentiment"], df["Count"], color=colors)
                    
                    # Add labels
                    ax.set_xlabel("Sentiment")
                    ax.set_ylabel("Number of Articles")
                    ax.set_title("Sentiment Distribution")
                    
                    # Add count labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.,
                            height + 0.1,
                            str(int(height)),
                            ha="center",
                            va="bottom"
                        )
                    
                    # Display plot
                    st.pyplot(fig)
                else:
                    st.write("No sentiment data available.")
            
            with col2:
                # Topic overlap
                st.subheader("Topic Analysis")
                topic_overlap = result["comparative_sentiment_score"].get("topic_overlap", {})
                
                st.write("**Common Topics:**")
                common_topics = topic_overlap.get("common_topics", [])
                if common_topics:
                    for topic in common_topics:
                        st.markdown(f"<span style='background-color:#E3F2FD;color:#1565C0;padding:5px 10px;border-radius:15px;margin-right:5px;font-size:0.8rem;'>{topic}</span>", unsafe_allow_html=True)
                else:
                    st.write("No common topics found.")
                
                st.write("**Unique Topics:**")
                unique_topics = topic_overlap.get("unique_topics", [])
                if unique_topics:
                    for topic in unique_topics[:10]:
                        st.markdown(f"<span style='background-color:#E3F2FD;color:#1565C0;padding:5px 10px;border-radius:15px;margin-right:5px;font-size:0.8rem;'>{topic}</span>", unsafe_allow_html=True)
                else:
                    st.write("No unique topics found.")
            
            # Coverage differences
            st.subheader("Coverage Differences")
            coverage_differences = result["comparative_sentiment_score"].get("coverage_differences", [])
            
            if coverage_differences:
                for diff in coverage_differences:
                    st.markdown(f"""
                    <div style='background-color:#f9f9f9;border-radius:10px;padding:20px;box-shadow:0 4px 6px rgba(0,0,0,0.1);margin-bottom:20px;'>
                        <p><strong>Comparison:</strong> {diff['comparison']}</p>
                        <p><strong>Impact:</strong> {diff['impact']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No significant coverage differences found.")
            
            # Final sentiment analysis
            st.subheader("Final Sentiment Analysis")
            st.markdown(f"""
            <div style='background-color:#f9f9f9;border-radius:10px;padding:20px;box-shadow:0 4px 6px rgba(0,0,0,0.1);margin-bottom:20px;'>
                <p>{result['final_sentiment_analysis']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Tab 3: Hindi TTS
        with tab3:
            # Get audio path
            audio_path = result.get("audio", "")
            
            if audio_path and os.path.exists(audio_path):
                # Display audio player
                st.subheader("Hindi TTS Summary")
                st.audio(audio_path)
                
                # Download button
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                
                st.download_button(
                    label="Download Hindi Audio",
                    data=audio_bytes,
                    file_name=f"{result['company']}_hindi_summary.mp3",
                    mime="audio/mpeg"
                )
            else:
                st.error("Audio file not found.")
    
    except Exception as e:
        # Display error
        st.error(f"Error: {str(e)}")
        progress_bar.empty()
else:
    # Display instructions
    st.info("Select a company and click 'Analyze News' to start the analysis.")
