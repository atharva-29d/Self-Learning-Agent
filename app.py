import streamlit as st

from main import chat, memory, USER_ID


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Local AI Memory Assistant",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# Custom styling
# ============================================================

st.markdown(
    """
    <style>

    .memory-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #444;
        margin-bottom: 10px;
    }

    .category-label {
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 5px;
    }

    .memory-text {
        font-size: 14px;
    }

    .score {
        font-size: 12px;
        color: #999;
        margin-top: 5px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Session state
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retrieved_memories" not in st.session_state:
    st.session_state.retrieved_memories = []

if "query_category" not in st.session_state:
    st.session_state.query_category = None


# ============================================================
# Get stored memories
# ============================================================

all_memories = memory.get_all(
    filters={"user_id": USER_ID}
)

stored_memories = all_memories.get("results", [])


# ============================================================
# Count categories
# ============================================================

category_counts = {
    "fact": 0,
    "preference": 0,
    "goal": 0,
    "project": 0,
    "skill": 0,
    "temporary": 0,
}

for item in stored_memories:

    metadata = item.get("metadata") or {}

    category = metadata.get("category")

    if category in category_counts:
        category_counts[category] += 1


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.title("🧠 Memory Dashboard")

    st.metric(
        "Total Memories",
        len(stored_memories)
    )

    st.divider()

    st.subheader("Memory Categories")

    st.write(
        f"📌 Facts: **{category_counts['fact']}**"
    )

    st.write(
        f"❤️ Preferences: **{category_counts['preference']}**"
    )

    st.write(
        f"🎯 Goals: **{category_counts['goal']}**"
    )

    st.write(
        f"📁 Projects: **{category_counts['project']}**"
    )

    st.write(
        f"🛠 Skills: **{category_counts['skill']}**"
    )

    st.write(
        f"🕒 Temporary: **{category_counts['temporary']}**"
    )

    st.divider()

    st.subheader("Stored Memories")

    for item in stored_memories:

        metadata = item.get("metadata") or {}

        category = metadata.get(
            "category",
            "uncategorized"
        )

        st.markdown(
            f"**{category.capitalize()}**"
        )

        st.caption(
            item["memory"]
        )

    st.divider()

    if st.button(
        "Clear Chat",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.session_state.retrieved_memories = []
        st.session_state.query_category = None

        st.rerun()


# ============================================================
# Main page
# ============================================================

st.title("🧠 Local AI Memory Assistant")

st.caption(
    "Persistent memory powered by "
    "Mem0, Qdrant, Llama 3.1 and Nomic embeddings."
)


# ============================================================
# Architecture information
# ============================================================

with st.expander("⚙️ System Information"):

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("LLM", "Llama 3.1")

    with col2:
        st.metric("Memory", "Mem0")

    with col3:
        st.metric("Vector DB", "Qdrant")

    with col4:
        st.metric("Embeddings", "Nomic 768-d")


# ============================================================
# Display chat history
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# ============================================================
# Retrieved memory panel
# ============================================================

if st.session_state.retrieved_memories:

    st.divider()

    st.subheader("🧠 Retrieved Memories")

    if st.session_state.query_category:

        st.caption(
            f"Query category: "
            f"**{st.session_state.query_category}**"
        )

    for item in st.session_state.retrieved_memories:

        metadata = item.get("metadata") or {}

        category = metadata.get(
            "category",
            "uncategorized"
        )

        score = item.get("score")

        col1, col2 = st.columns(
            [5, 1]
        )

        with col1:

            st.markdown(
                f"**{category.capitalize()}**"
            )

            st.write(
                item["memory"]
            )

        with col2:

            if score is not None:

                st.metric(
                    "Score",
                    f"{score:.3f}"
                )

        st.divider()


# ============================================================
# Chat input
# ============================================================

user_message = st.chat_input(
    "Ask something about yourself..."
)


if user_message:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    with st.chat_message("user"):

        st.markdown(
            user_message
        )

    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching memory and generating response..."
        ):

            answer, retrieved_memories = chat(
                user_message
            )

        st.markdown(answer)

    # --------------------------------------------------------
    # Save assistant response
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    # --------------------------------------------------------
    # Save retrieved memories
    # --------------------------------------------------------

    st.session_state.retrieved_memories = (
        retrieved_memories
    )

    # --------------------------------------------------------
    # Query category
    #
    # We already print this in main.py, but we need it
    # available in the UI.
    # --------------------------------------------------------

    # The retrieved memories themselves demonstrate the
    # category-aware search. We can infer the displayed
    # category from the retrieved results for now.

    if retrieved_memories:

        metadata = (
            retrieved_memories[0].get("metadata")
            or {}
        )

        st.session_state.query_category = (
            metadata.get("category")
        )

    else:

        st.session_state.query_category = "all"

    st.rerun()