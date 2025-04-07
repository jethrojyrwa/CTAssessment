import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Load BERT model
@st.cache_resource
def load_bert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load Questions
@st.cache_data
def load_questions_from_csv(file_path):
    df = pd.read_csv(file_path)
    question_bank = {}
    for _, row in df.iterrows():
        concept = row["concept"]
        question_data = {
            "question": row["question"],
            "options": [row["option1"], row["option2"], row["option3"], row["option4"]],
            "correct": row["correct"]
        }
        question_bank.setdefault(concept, []).append(question_data)
    return question_bank

# Calculate BERT Similarity
def compute_bert_similarity(user_answer, correct_answer):
    user_embedding = bert_model.encode([user_answer])
    correct_embedding = bert_model.encode([correct_answer])
    return cosine_similarity(user_embedding, correct_embedding)[0][0]

# Score Calculator
def compute_score(concept, responses, question_bank):
    total_similarity = 0
    questions = question_bank[concept]
    for i, user_answer in enumerate(responses):
        correct_answer = questions[i]['correct']
        total_similarity += compute_bert_similarity(user_answer, correct_answer)
    return round(total_similarity / len(responses), 3)

# Feedback Definitions
concept_feedback = {
    "Decomposition": {
        "excellent": "You have demonstrated an outstanding ability to break down complex problems into smaller, manageable parts.",
        "good": "You can decompose problems well, but refining your breakdown process will enhance efficiency.",
        "fair": "You have a basic understanding of decomposition, but try to identify more granular components in a problem.",
        "needs_improvement": "Your decomposition approach needs work. Focus on dividing problems into distinct subproblems for better clarity."
    },
    "Pattern Recognition": {
        "excellent": "Great job identifying patterns! Your ability to recognize recurring structures helps in efficient problem-solving.",
        "good": "You are able to recognize patterns, but improving consistency in identifying key patterns will help further.",
        "fair": "You understand basic patterns, but need to improve in applying them effectively to problem-solving.",
        "needs_improvement": "Struggled with recognizing patterns. Try practicing problems that require finding similarities in solutions."
    },
    "Abstraction": {
        "excellent": "Excellent abstraction skills! You can filter out unnecessary details and focus on key principles.",
        "good": "Good abstraction! Work on simplifying problems even further by removing irrelevant details.",
        "fair": "You understand abstraction but sometimes retain unnecessary details. Try focusing on core aspects.",
        "needs_improvement": "Struggled with abstraction. Work on identifying key problem aspects while discarding irrelevant details."
    },
    "Algorithmic Thinking": {
        "excellent": "You have a strong grasp of algorithmic thinking, demonstrating efficiency in stepwise problem-solving.",
        "good": "Good algorithmic thinking! A bit more precision in structuring your steps will improve your performance.",
        "fair": "Fair understanding of algorithmic processes, but try to refine the logical flow of your solutions.",
        "needs_improvement": "Your algorithmic thinking needs improvement. Try practicing structured approaches to problem-solving."
    }
}

def get_feedback(score):
    if score >= 0.8:
        return "Excellent", "excellent"
    elif score >= 0.6:
        return "Good", "good"
    elif score >= 0.4:
        return "Fair", "fair"
    else:
        return "Needs Improvement", "needs_improvement"

# UI Setup
st.set_page_config(page_title="CT Assessment", layout="centered")

tab1, tab2 = st.tabs(["🧠 Assessment", "📈 Data Analysis"])
with tab1:

    st.title("🧠 Computational Thinking Assessment")

    # Load model and questions
    bert_model = load_bert_model()
    csv_file_path = "computational_thinking_questions.csv"
    question_bank = load_questions_from_csv(csv_file_path)
    concepts = ['Decomposition', 'Pattern Recognition', 'Abstraction', 'Algorithmic Thinking']

    # Collect demographic details first
    if 'user_details_collected' not in st.session_state:
        with st.form("user_details"):
            st.subheader("👤 Participant Information")
            course = st.radio("Course:", ["BCA", "MCA", "MSc", "BSc", "PhD"])
            major = st.text_input("If pursuing BSc or MSc or Others, please specify your major:")
            age_group = st.radio("Age:", ["16 - 19", "20 - 23", "24 - 27", "28 - 31", "Above 31"])
            sex = st.radio("Sex:", ["Male", "Female", "Prefer not to say"])

            if st.form_submit_button("Start Assessment"):
                st.session_state.course = course
                st.session_state.major = major.strip() if major and major.strip() != "" else None
                st.session_state.age_group = age_group
                st.session_state.sex = sex
                st.session_state.user_details_collected = True
                st.rerun()

    # Initialize state
    if 'scores' not in st.session_state:
        st.session_state.scores = {}
    if 'answers' not in st.session_state:
        st.session_state.answers = {}

    # Ask questions after user details
    if 'user_details_collected' in st.session_state:
        for concept in concepts:
            if concept not in st.session_state.scores:
                with st.form(key=f"{concept}_form"):
                    st.header(f"{concept} Questions")
                    responses = []
                    for i, q in enumerate(question_bank[concept]):
                        response = st.radio(f"Q{i+1}: {q['question']}", q['options'], key=f"{concept}_q{i}")
                        responses.append(response)

                    if st.form_submit_button("Submit"):
                        score = compute_score(concept, responses, question_bank)
                        st.session_state.scores[concept] = score
                        st.session_state.answers[concept] = responses
                        label, key = get_feedback(score)
                        st.success(f"{concept} Score: {score} → {label}")
                        st.info(concept_feedback[concept][key])

                        # Auto-save logic when all concepts completed
                        if len(st.session_state.scores) == 4 and 'saved' not in st.session_state:
                            try:
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                row_data = {
                                    "Timestamp": timestamp,
                                    "Course": st.session_state.course,
                                    "Major": st.session_state.major,
                                    "Age Group": st.session_state.age_group,
                                    "Sex": st.session_state.sex,
                                }
                                for c in concepts:
                                    for j, answer in enumerate(st.session_state.answers[c]):
                                        question_text = question_bank[c][j]['question']
                                        row_data[question_text] = answer
                                avg_score = round(sum(st.session_state.scores.values()) / 4, 3)
                                perf_label, _ = get_feedback(avg_score)
                                row_data.update({
                                    "Decomposition Score": st.session_state.scores["Decomposition"],
                                    "Pattern Recognition Score": st.session_state.scores["Pattern Recognition"],
                                    "Abstraction Score": st.session_state.scores["Abstraction"],
                                    "Algorithmic Thinking Score": st.session_state.scores["Algorithmic Thinking"],
                                    "CT Overall Score": avg_score,
                                    "Performance Category": perf_label
                                })

                                df = pd.DataFrame([row_data])
                                file_path = "responses_scored.csv"
                                df.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))

                                st.toast("✅ Auto-saved your responses to 'responses_scored.csv'")
                                st.session_state.saved = True

                            except Exception as e:
                                st.error(f"❌ Error auto-saving results: {e}")

        # Summary
        if len(st.session_state.scores) == 4:
            st.subheader("✅ Summary Result")
            avg_score = round(sum(st.session_state.scores.values()) / 4, 3)
            label, _ = get_feedback(avg_score)
            st.write("### 💡 Overall CT Score:", avg_score)
            st.write("### 🏆 Performance Category:", label)
            st.divider()

            for concept in concepts:
                score = st.session_state.scores[concept]
                label, key = get_feedback(score)
                st.write(f"**{concept} Score:** {score} → {label}")
                st.caption(concept_feedback[concept][key])

        if st.button("🔁 Retake Assessment"):
            for concept in concepts:
                for i in range(len(question_bank[concept])):
                    q_key = f"{concept}_q{i}"
                    if q_key in st.session_state:
                        del st.session_state[q_key]
            for key in ['scores', 'answers', 'user_details_collected', 'saved']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()  

with tab2:
    st.title("📊 CT Assessment Data Analysis")

    try:
        df = pd.read_csv("responses_scored.csv")
        st.success("✅ Loaded assessment data")

        st.dataframe(df.head())

        st.write("Total Numbre of Samples: ", len(df))
        subset_df = df[['Course', 'If pursuing BSc or MSc or Others, please specify your major', 'Age', 'Sex', 'Decomposition Score', 'Pattern Recognition Score', 'Abstraction Score', 'Algorithmic Thinking Score', 'CT Overall Score', 'Performance Category']]
        
        # Distribution of CT Overall Scores
        fig = plt.figure(figsize=(10, 6))
        sns.histplot(subset_df['CT Overall Score'], kde=True)
        plt.title('Distribution of CT Overall Scores')
        plt.xlabel('CT Overall Score')
        plt.ylabel('Frequency')
        st.pyplot(fig)
        
        
        # Boxplot of CT Overall Score by Performance Category
        fig = plt.figure(figsize=(10, 6))
        sns.boxplot(x='Performance Category', y='CT Overall Score', data=subset_df)
        plt.title('CT Overall Score by Performance Category')
        plt.xlabel('Performance Category')
        plt.ylabel('CT Overall Score')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Pairplot of relevant scores
        fig = plt.figure(figsize=(10, 6))
        sns.pairplot(subset_df[['Decomposition Score', 'Pattern Recognition Score', 'Abstraction Score', 'Algorithmic Thinking Score', 'CT Overall Score']], diag_kind='kde')
        plt.suptitle('Pairwise Relationships of CT Sub-Scores', y=1.02)
        st.pyplot(fig)
        
        # Correlation Heatmap
        fig = plt.figure(figsize=(10, 8))
        correlation_matrix = subset_df[['Decomposition Score', 'Pattern Recognition Score', 'Abstraction Score', 'Algorithmic Thinking Score', 'CT Overall Score']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix of CT Sub-Scores')
        st.pyplot(fig)

        fig = plt.figure(figsize=(12, 6))
        sns.barplot(x='Course', y='CT Overall Score', data=subset_df)
        plt.title('Average CT Overall Score by Course')
        plt.xlabel('Course')
        plt.ylabel('Average CT Overall Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout() # Prevents labels from overlapping
        st.pyplot(fig)

        # Min-Max normalization
        from sklearn.preprocessing import MinMaxScaler

        # Create a MinMaxScaler object
        scaler = MinMaxScaler()

        # Fit the scaler to the 'CT Overall Score' column and transform it
        subset_df['CT Overall Score Normalized'] = scaler.fit_transform(subset_df[['CT Overall Score']])
        average_normalized_scores = subset_df.groupby('Course')['CT Overall Score Normalized'].mean()

        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'orange', 'purple'] # Customize colors as needed
        fig = plt.figure(figsize=(12, 6))
        ax = average_normalized_scores.plot(kind='bar', figsize=(10, 6), color=colors)

        # Add labels and title
       

        plt.title('Average Normalized CT Overall Score by Course')
        plt.xlabel('Course')
        plt.ylabel('Average Normalized CT Overall Score')
        plt.xticks(rotation=0) # Rotate x-axis labels if needed

        # Customize the plot (optional)
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Failed to load or analyze data: {e}")
