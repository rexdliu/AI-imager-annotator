"""Card-based QA pair selector component."""

from typing import List, Callable, Optional

import streamlit as st

from utils.gemini_api import ClaudeQA
from utils.logger import get_logger

logger = get_logger("qa_selector")


def qa_card_selector(qa_pairs: List[ClaudeQA], on_select_callback: Callable[[ClaudeQA], None]) -> None:
    """
    Display QA pairs as horizontal cards with individual selection buttons.

    Args:
        qa_pairs: List of ClaudeQA objects to display
        on_select_callback: Callback function that takes the selected ClaudeQA
    """
    if not qa_pairs:
        st.error("No QA pairs available to select from.")
        return

    st.subheader("🤖 Select a Question-Answer Pair")
    st.info("Choose one of the AI-generated QA pairs to add to your annotation:")

    # Add custom CSS for card layout
    st.markdown("""
    <style>
    .qa-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        background-color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        position: relative;
    }
    .qa-card:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .qa-header {
        font-weight: bold;
        border-bottom: 1px solid #eee;
        padding-bottom: 10px;
        margin-bottom: 10px;
    }
    .qa-score {
        position: absolute;
        top: 15px;
        right: 15px;
        background-color: #f8f9fa;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
    }
    .qa-easy { color: #28a745; }
    .qa-medium { color: #fd7e14; }
    .qa-hard { color: #dc3545; }
    </style>
    """, unsafe_allow_html=True)

    # Determine number of columns based on number of QA pairs
    num_cols = min(3, len(qa_pairs))
    if num_cols == 0:
        return

    # Create columns for layout
    cols = st.columns(num_cols)

    # Distribute cards among columns
    for i, qa in enumerate(qa_pairs):
        col_idx = i % num_cols

        with cols[col_idx]:
            # Determine difficulty color
            difficulty_class = qa.difficulty
            difficulty_color = {
                "easy": "#28a745",
                "medium": "#fd7e14",
                "hard": "#dc3545"
            }.get(difficulty_class, "#333333")

            # Quality score coloring
            score = qa.language_quality_score
            score_color = "green" if score > 3.5 else "orange" if score > 2 else "red"

            # Create card with styled elements
            st.markdown(f"""
            <div class="qa-card">
                <div class="qa-score" style="color: {score_color};">
                    {score}
                </div>
                <div class="qa-header">
                    {qa.task_type.upper()} <span style="color: {difficulty_color};">({difficulty_class})</span>
                </div>
                <div style="font-size: 0.9em; margin-bottom: 10px;">
                    <div><strong>🇬🇧 Q:</strong> {qa.text_en[:100] + '...' if len(qa.text_en) > 100 else qa.text_en}</div>
                    <div><strong>🌐 Q:</strong> {qa.text_local[:100] + '...' if len(qa.text_local) > 100 else qa.text_local}</div>
                </div>
                <div style="font-size: 0.9em; margin-bottom: 10px;">
                    <div><strong>🇬🇧 A:</strong> {qa.answer_en[:100] + '...' if len(qa.answer_en) > 100 else qa.answer_en}</div>
                    <div><strong>🌐 A:</strong> {qa.answer_local[:100] + '...' if len(qa.answer_local) > 100 else qa.answer_local}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Button to select this QA pair
            if st.button(f"Use This QA", key=f"use_qa_{i}", use_container_width=True):
                logger.info(f"Selected QA pair {i + 1}: {qa.task_type} ({qa.difficulty})")
                on_select_callback(qa)

    # Add detailed view of each QA pair in expandable sections
    for i, qa in enumerate(qa_pairs):
        with st.expander(f"Details for QA #{i + 1}: {qa.task_type.upper()} ({qa.difficulty})"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### English")
                st.markdown(f"**Question:** {qa.text_en}")
                st.markdown(f"**Answer:** {qa.answer_en}")

            with col2:
                st.markdown("#### Local Language")
                st.markdown(f"**Question:** {qa.text_local}")
                st.markdown(f"**Answer:** {qa.answer_local}")

            st.markdown(f"**Tags:** {', '.join(qa.tags) if qa.tags else 'None'}")
            st.markdown(f"**Quality Score:** {qa.language_quality_score}/5")