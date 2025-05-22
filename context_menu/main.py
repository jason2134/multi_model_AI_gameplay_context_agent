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

    # Select last N entries
    test_prompt = df[['timestamp_ms', 'events', 'combinedContext']].tail(30)

    # Preprocess into readable format
    readable_prompt = preprocess_logs(test_prompt)

    # Initialize model
    model = OllamaLLM(model='llama3.2')

    # Create new structured prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """You are an expert at analyzing real-time game data and giving advice to players. 
         
Input data: The input given is originated from a combined data from OCR, Yolo, and game textual logs. The following is the meaning of each column:
- timestamp: 
    - Current time when a certain event occured.
- player_ammo: 
    - Current player ammo, this value could only be 100 at most. Ammo is considered low if it is less than 10
- player_health: 
    - Current health of the player. 
    - Health can only be at most 100. 
    - If more health is picked it could get upto 200.When it reaches 200 it will gradullay decrease until it reaches 100 again.
    - If player health is 0 while a killed or suicide event happens on the same line. This means the player is dead. 
- player_armor:
    - Current armor value the player has
    - Armor can only be at most 100. 
    - If more health is picked it could get upto 200.When it reaches 200 it will gradullay decrease until it reaches 100 again. 
    
*** if player armor, health, and ammo are all zero while there is no killed or suicide event, that means the player is not in the game, and cannot be analyzed
         
         
         
         
         
Tasks:
1. Summarize key events (enemy sightings, health/ammo/armour status) based only on HIGH confidence information (>60%).
2. Give practical advice based on the situation (e.g., low health -> find health packs, enemy detected -> prepare for combat).
         
Rules:
- ONLY refer to detections with confidence higher than 0.6.
- Give short and clear advice players can act on immediately.
- If ammo is low, say "Find ammo."
- If armor is low, say "Find armor."
- If health is low, say "Seek health immediately."
- If enemies detected, warn about enemy positions.
- Note that the maximum number for ammo, health, and armor can only be 100. If more health and armor is picked it could get upto 200. When it reaches 200 it will gradullay decrease until it reaches 100 again

When outputing the advice, pelase keep it short:
- If ammo is low, say "Find ammo."
- If armor is low, say "Find armor."
- If health is low, say "Seek health immediately."
- If enemy found, say "Prepare for combat."
- No need to explain more, keep each type of advice in point form and less than 5 words
"""),
        ("human", 
         "Analyze the following log information and provide a summary and advice on next actions:\n{info}")
    ])

    # Chain
    chain = prompt | model

    # Invoke
    result = chain.invoke({"info": readable_prompt})

    end_time = time.time()

    # Output
    print(result)
    print(f"Execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()



# 10 --> 2 seconds