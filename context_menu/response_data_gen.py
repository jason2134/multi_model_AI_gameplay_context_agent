import pandas as pd
import time
import ast
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

def preprocess_logs(df):
    """Preprocess the context logs into a human-readable summary for the model."""
    summaries = []
    
    for _, row in df.iterrows():
        combined_context = row['combinedContext']
        
        # Convert string representation of dict to real dict
        context = ast.literal_eval(combined_context)

        entry = []
        
        if 'yolo' in context:
            for obj in context['yolo']:
                if obj['confidence'] > 0.6:  # only keep high confidence detections
                    entry.append(f"Enemy detected at ({obj['center_x']:.1f}, {obj['center_y']:.1f}) with confidence {obj['confidence']:.2f}")
        
        if 'ocr' in context:
            for obj in context['ocr']:
                if obj['conf_roi_ammo'] > 0.6:
                    entry.append(f"Ammo: {obj['pred_roi_ammo']} (confidence {obj['conf_roi_ammo']:.2f})")
                if obj['conf_roi_health'] > 0.6:
                    entry.append(f"Health: {obj['pred_roi_health']} (confidence {obj['conf_roi_health']:.2f})")
                if obj['conf_roi_armour'] > 0.6:
                    entry.append(f"Armour: {obj['pred_roi_armour']} (confidence {obj['conf_roi_armour']:.2f})")
        
        if entry:
            summaries.append(f"Timestamp {row['timestamp_ms']}: " + "; ".join(entry))
    
    return "\n".join(summaries)

def main():
    start_time = time.time()

    # Load data
    df = pd.read_csv("../data_processors/data/processed/final/contextMenu_0ms.csv")

    # Initialize model
    model = OllamaLLM(model='llama3.2')

    # Create new structured prompt
    prompt = ChatPromptTemplate.from_messages([(
        "system", 
        """You are an expert in analyzing Quake 3 real-time game data and advising players. Your task is to:

            1. Summarize key events based ONLY on HIGH confidence info (>60%).
            2. Provide practical, actionable advice.
            3. Talk in a tone that you are advising the player when he / she is watching the playback from the game, don't say something like: according to the log. Use natural language as much as possible and avoid terms that are too technical

            🧠 RULES (Follow strictly):
            - Summarize enemy sightings and item sightings (weapon, ammo, armor, health), player's health, ammo, and armor status.
            - Only report detections with confidence > 60%.
            - No speculation or explanation. Use only available data.
            - Consider health < 30 as low.
            - Consider ammo < 10 as low.
            - Max value for health, ammo, and armor is 100 (can go up to 200 temporarily).
            - If health is missing, the player is dead.
            - If nothing is detected for ammo/armor, the player has none.
            - If enemy appears repeatedly, assume same enemy spotted multiple times.
            - Do not refer to yourself, your role, or the task.
            - Enemy health, armor, and ammo are always unknown, and do not need to mention in the response
            - 📏 Word Limit: Keep total under 40 words. Focus on clarity and brevity.

            📝 OUTPUT FORMAT (Strictly follow this. DO NOT ADD ANYTHING ELSE):

            **Summary:**
            * Bullet 1
            * Bullet 2
            ...

            **Advice:**
            * Bullet 1
            * Bullet 2
            ...

            ❗️Output MUST contain ONLY the two above sections — no headers, no explanations, no notes, and no additional text.

            📌 Important Phrases (Use exactly as written):
            - "Find ammo."
            - "Find armor."
            - "Seek health immediately."
            - "Prepare for combat."
        """),
        ("human", 
         "Analyze the following log information and provide a summary and advice on next actions:\n{info}")
    ])

    # Prepare dataframe to store results
    responses = []
    response_nums = []

    # Loop over the dataframe in chunks of 10 lines
    chunk_size = 10
    total_lines = len(df)
    for i in range(0, total_lines, chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        
        # Preprocess the chunk into readable format
        readable_prompt = preprocess_logs(chunk)
        
        # Chain the prompt with the model
        chain = prompt | model

        # Invoke the model and get the result
        result = chain.invoke({"info": readable_prompt})
        
        print(result)
        
        # Store the response with the corresponding response number
        responses.append(result)
        response_nums.append(i // chunk_size + 1)  # Response number is based on the chunk index

    # Create dataframe from the responses
    result_df = pd.DataFrame({
        'response_num': response_nums,
        'response': responses
    })

    # Save the dataframe to a CSV file
    result_df.to_csv('generated_responses.csv', index=False)

    end_time = time.time()

    # Output
    print(f"Execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()