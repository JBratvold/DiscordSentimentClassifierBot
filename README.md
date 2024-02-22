# Discord Sentiment Classifier Bot
 This Discord bot implements a sentiment analysis functionality using a variant of the Naive Bayes machine learning algorithm, allowing users to train the bot with labeled data for improved accuracy in classifying the sentiment of text-based inputs within the Discord platform.

## Overview
Primary purpose is to determine if a given text has either a positive or negative sentiment.
The Python script implements a Discord bot using primarily the "discord.ext" library. The bot can be trained with user-provided data to improve its classification accuracy.


## Requirements
- stable internet connection (to connect the bot to your discord server)
- discord server (where you can access the channel id and have permission to add a bot)
- Python 3.x
- discord.py library
- nltk library
- other (asyncio, re, string)


## Installation
1. Clone the repository: "git clone https://github.com/JBratvold/DiscordSentimentClassifierBot.git"
2. Install required libraries.
3. Set up a Discord bot and obtain the token. This is done through discord developers portal.
4. Replace the placeholder DISCORD_TOKEN ("sample_token_1234567") in the script with your bot's token.
5. Replace the placeholder DISCORD_CHANNEL_ID (1) in the script with your discord servers channel id in which the bot will interact with.
5. Adjust other constants and file paths as needed.

## Usage
1. Run the script: "python3 DiscordSentimentAnalyzerBot.py"
2. Invite the bot to your Discord server.
3. Use the "!train" command to add training data for the bot. The bot will prompt you to classify the sentiment of the provided comment using reactions.
4. Use the "!classify" command to classify a comment without adding it to the training data.

## Discord Commands
- !train [comment]         Add training data for the bot. The bot will prompt you to classify the sentiment of the provided comment.
- !classify [comment]      Classify a comment using the bot's sentiment analysis.

## Files/Images
- "stopwords.txt": List of stopwords used for filtering.
- "posTrainingData.txt": Positive training data.
- "negTrainingData.txt": Negative training data.
- "checkmark.png": Image containing a green checkmark
- "smiley.png": Image containing a smiley face
- "frowney.png": Image containing a frowning face

## Script Layout
The python script is written and organized in the following sections:
- Imports
- Constants
- Functions
- Async Functions
- Discord Events
- Discord Commands
- Discord Bot Start-Up
The function calls are initiated on bot start-up, and then are triggered as the user inputs a command to the bot through discord.