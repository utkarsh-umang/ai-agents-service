from langchain_core.prompts import ChatPromptTemplate

thumbnail_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert YouTube thumbnail designer. {system_prompt}",
        ),
        (
            "human",
            """Analyze the provided images and design brief, then describe
        a detailed DALL-E prompt to generate a thumbnail.

        Title: {title}
        Include title in thumbnail: {include_title}
        Creative direction: {creative_comments}

        Describe composition, colors, style, and mood in detail.""",
        ),
    ]
)
