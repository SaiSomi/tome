import random

# Master list of unique notes (no topic headers, one of each)
notes = [
    # SPY
    "SPY rallied after CPI data came in below expectations.",
    "Investors rotated into SPY ETFs following dovish Fed signals.",
    "SPY options saw record volume ahead of Powell's speech.",
    "The SPY broke resistance at 450 after a strong earnings season.",

    # TSLA
    "TSLA missed earnings estimates due to lower margins.",
    "Tesla’s delivery numbers surprised analysts in Q2.",
    "TSLA stock surged after announcing a new battery factory.",
    "Elon Musk discussed Tesla's AI plans during the earnings call.",

    # Fed
    "The Fed is expected to pause rate hikes amid mixed inflation data.",
    "Powell signaled a data-dependent approach in the coming months.",
    "FOMC minutes suggest continued caution on inflation persistence.",
    "Rate hikes have begun to cool down the housing and job markets.",

    # Options
    "Call option flows increased sharply in SPY and QQQ.",
    "Retail traders are piling into weekly TSLA calls.",
    "Unusual options activity spotted before CPI release.",
    "Put-call ratio hit a 3-month low amid bullish sentiment.",

    # Earnings
    "Apple beat earnings expectations driven by services growth.",
    "Amazon posted strong cloud revenue despite macro headwinds.",
    "Bank stocks dropped following weak earnings and loan loss reserves.",
    "Tech earnings led to major sector ETF inflows.",

    # Commodities
    "Oil prices rose on Middle East tensions and supply concerns.",
    "Gold climbed as inflation expectations increased.",
    "WTI crude pulled back after inventory data showed a surplus.",
    "Silver outperformed gold in recent commodity rallies.",

    # Macro
    "Unemployment claims rose unexpectedly this week.",
    "CPI rose 3.2% YoY, slightly above estimates.",
    "GDP growth came in stronger than expected at 2.4%.",
    "Consumer sentiment dipped amid rising credit card delinquencies.",

    # AI + Markets
    "AI models are increasingly being used to predict market movements.",
    "Hedge funds are integrating LLMs into trading strategies.",
    "Retail investors are using AI tools to filter earnings reports.",
    "AI-related stocks like NVDA and AMD are leading the rally."
]

# Optional variation: add human-like noise
for i in range(len(notes)):
    if random.random() < 0.3:
        notes[i] += " Analysts remain divided on what this means for the broader market."
    if random.random() < 0.2:
        notes[i] += " More updates are expected next week."

# Shuffle the notes to make graph layout less linear
random.shuffle(notes)

# Write to notes.txt with double newlines
with open("notes.txt", "w") as f:
    f.write("\n\n".join(notes))

print(f"✅ Written {len(notes)} unique, shuffled notes to notes.txt")
