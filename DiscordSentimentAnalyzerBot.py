"""
README

# DiscordSentimentAnalyzerBot

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
1. Clone the repository: "git clone https://github.com/PrivateAccount.git"
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


## Author
Joshua Bratvold

"""

##################################################
#                  IMPORTS                       #
##################################################
import asyncio
import discord
import re as RegularExpression
import string
from nltk.stem import PorterStemmer
from discord.ext import commands

##################################################
#                  CONSTANTS                     #
##################################################



# Token/ID's for discord 
# TODO - Update token with your discord bots valid token
DISCORD_TOKEN = "sample_token_1234567"



# Channel ID that the bot will interact in 
# TODO - Update Channel Id with your discord servers channel id
DISCORD_CHANNEL_ID = 1 



# Create an instance of the bot
INTENTS = discord.Intents.all()
BOT_PREFIX = "!"
DISCORD_BOT = commands.Bot(command_prefix=BOT_PREFIX,intents=INTENTS)

# Colours
COLOUR_GREEN = 0x00ff00
COLOUR_RED = 0xff073a

# Emoji's
EMOJI_CROSSMARK = "\U0000274C" # Red "X"
EMOJI_SMILE = "\U0001F642" # Smiley Face
EMOJI_FROWN = "\U0001F641" # Frown Face

# Discord Messages
PROMPT_SELECT_SENTIMENT = "Which sentiment is most accurate?"
MSG_TITLE_DATA_ADDED = f"**Training successful!**"
MSG_CLASSIFICATION_OPTIONS = f"**Positive** {EMOJI_SMILE} \t**Negative** {EMOJI_FROWN} \t**Cancel** {EMOJI_CROSSMARK}"

# Images
PATH_TO_IMAGES = "img/"
IMG_CHECKMARK = "checkmark.png" 
IMG_POSITIVE = "smiley.png"
IMG_NEGATIVE = "frowney.png"

# Files
STOPWORDS_DATA = "data/stopwords.txt"
POS_TRAINING_DATA = "data/posTrainingData.txt"
NEG_TRAINING_DATA = "data/negTrainingData.txt"

# Global Variables
base_pos_probability = 0
base_neg_probability = 0
stopwords = ""
posWordList = ""
negWordList = ""
posProbabilities = ""
negProbabilities = ""

##################################################
#                  FUNCTIONS                     #
##################################################
def convertFileToList(filePath: str) -> list:
    """
    Reads a file and returns a list containing each line as an element.

    Parameters:
    - file (str): The path to the file to be read.

    Returns:
    list: A list containing each line of the file as a string, with leading and trailing whitespaces removed.

    Example:
    >>> convertFileToList("example.txt") 
    ['Line 1', 'Line 2', 'Line 3']
    """
    f = open(filePath, "r")
    list = [line.strip() for line in f]
    f.close()
    return list

def parseMessage(dirtyString: str) -> list:
    """
    Process a dirty (non parsed/filtered) string to extract a clean (processed) list of words.

    Parameters:
    - dirtyString (str): The input string to be processed.

    Returns:
    list: A list of clean, stemmed, and filtered words.

    Example:
    >>> parseMessage("This is A weird! meSsagE with punctuations @nd hyperlinks??")
    ['weird', 'messag', 'punctuat', 'hyperlink']
    """
    # Remove punctuations
    dirtyString = removePunctuationFromString(dirtyString)

    # Remove hyperlinks
    dirtyString = RegularExpression.sub(r'https?:\/\/.*[\r\n]*', '', dirtyString)

    # Make lowercase and split into a list of individual words
    dirtyListOfWords = dirtyString.lower().split()

    # Stem every word in the list
    stemmedListOfWords = stemList(dirtyListOfWords)

    # Filter the list removing all stopwords
    cleanListOfWords = filterStopwords(stemmedListOfWords, stopwords)

    return cleanListOfWords

def filterStopwords(wordsToFilter: list, stopwords: list) -> list:
    """
    Filter out stopwords from a list of words.

    (Stopwords: "Stop words are the words in a stop list (or stoplist or negative dictionary) which are filtered out (i.e. stopped)
      before or after processing of natural language data (text) because they are insignificant. Ex: The, a, it, etc...")

    Parameters:
    - wordsToFilter (list): List of words to run the filter on.
    - stopwords (list): List containing stopwords (words to remove).

    Returns:
    list: A filtered list of words that don't contain any of the words found in stopwords.

    Example:
    >>> filterStopwords(['this', 'is', 'a', 'sample', 'sentence'], ['is', 'a'])
    ['this', 'sample', 'sentence']
    """
    return [word for word in wordsToFilter if word not in stopwords]

def stemList(listToStem: list) -> list:
    """
    Perform stemming on a list of words using the Porter Stemmer algorithm from nltk.stem import PorterStemmer

    Parameters:
    - listToStem (list): A list of words to undergo stemming.

    Returns:
    list: A list containing the stemmed forms of the input words.

    Example:
    >>> stemList(['running', 'easily', 'unhappiness'])
    ['run', 'easili', 'unhappi']
    """
    return [PorterStemmer().stem(word) for word in listToStem]

def removePunctuationFromString(stringToRemovePunctuation: str) -> str:
    """
    Removes all punctuation from an inputted string. The punctuations it removes include all the following symbols: 
    ! # " $ % & ' ( ) * + , - . / : ; < = > ? @ [ \ ] ^ _ ` { | } ~

    Parameters:
    - stringToRemovePunctuation (str): The input string from which punctuation will be removed.

    Returns:
    str: A new string with all punctuation removed.

    Example:
    >>> removePunctuationFromString("Hello, this is a sample text!!")
    'Hello this is a sample text'
    """
    # Make a translator object to replace punctuation with none
    translator = str.maketrans('', '', string.punctuation)

    # Use the translator
    return stringToRemovePunctuation.translate(translator)

def combineListIntoSingleString(commentList: list) -> str:
    """
    Combines a list of comments into a single string.

    Parameters:
    - commentList (list): A list of comments, where each comment is a string.

    Returns:
    str: A single string containing all comments concatenated with spaces.

    Example:
    >>> combineListIntoSingleString(['Great work!', 'Keep it up.'])
    'Great work! Keep it up.'
    """
    comments = ' '.join(commentList)
    return comments

def calculateFrequencies(trainingDataWordList: list) -> dict:
    """
    Counts the number of times a word is found in the list and assigns it a frequency.

    Parameters:
    - trainingDataWordList (list): A list of words.

    Returns:
    wordFrequenciesDictionary (dictionary): A dictionary where keys are unique words from the list, and values are the frequency of each word.

    Example:
    >>> calculateFrequencies(['apple', 'banana', 'apple', 'orange', 'banana'])
    {'apple': 2, 'banana': 2, 'orange': 1}
    """
    # Creating an empty dictionary to store word frequencies
    wordFrequenciesDictionary = {}
    
    # Count the frequency of each word in the list
    for word in trainingDataWordList:
        # When the word is not in the dictionary, set its count to 0; then, increment the count by 1, otherwise increment its count by 1
        wordFrequenciesDictionary[word] = wordFrequenciesDictionary.get(word, 0) + 1

    return wordFrequenciesDictionary

def calculateWordProbability(wordFrequencyDictionary: dict, totalWords: int) -> dict:
    """
    Calculates the probability of each word in a dictionary based on its count.

    Parameters:
    - wordFrequencyDictionary (dict): A dictionary where keys are words and values are their counts.
    - totalWords (int): The total number of words in the dataset.

    Returns:
    dict: A dictionary where keys are words, and values are their probabilities based on their counts.

    Example:
    >>> calculateProbabilityOfWord({'apple': 2, 'banana': 1, 'orange': 3}, 6)
    {'apple': 1/3, 'banana': 1/6, 'orange': 1/2}
    """
    # Creating an empty dictionary to store word probabilities
    probabilitiesDictionary = {}

    # Calculating the probability of each word
    for word, count in wordFrequencyDictionary.items():
        probabilitiesDictionary[word] = count / totalWords

    return probabilitiesDictionary

def processData(filePath: str) -> list:
    """
    Process data from a file by converting it to the correct format, combining into a single string, and parsing unwanted texts.

    Parameters:
    - filePath (str): The path to the file containing the data.

    Returns:
    list: A processed list of strings after removing unwanted elements.

    Example:
    >>> processData("data.txt")
    ['processed', 'data', 'output']
    """
    # Convert file to a list of strings
    listToConvertToString = convertFileToList(filePath)
    
    # Combine the list into a single string
    dataStringToProcess = combineListIntoSingleString(listToConvertToString)
    
    # Parse the combined string to remove unwanted elements
    processedData = parseMessage(dataStringToProcess)
    
    return processedData

def getWordCount(wordList: list) -> int:
    """
    Returns the number of words found in a list.

    Parameters:
    - wordList (list): A list of words.

    Returns:
    int: The number of words in the provided list.

    Example:
    >>> getWordCount(['apple', 'banana', 'orange'])
    3
    """
    return len(wordList)

def calculateSentimentProbabilities(trainingDataWordList: list) -> dict:
    """
    Calculate the probabilities of words in the training data and return a dictionary of word probabilities.

    Parameters:
    - trainingDataWordList (list): A list of words from the training data.

    Returns:
    dict: A dictionary where keys are words, and values are their probabilities based on their frequencies.

    Example:
    >>> calculateSentimentProbabilities(['apple', 'banana', 'apple', 'orange', 'banana'])
    {'apple': 2/5, 'banana': 2/5, 'orange': 1/5}
    """

    # Calculate word frequencies in the training data
    frequenciesDictionary = calculateFrequencies(trainingDataWordList)
    
    # Calculate the probability of each word based on its frequency
    probabilitiesDictionary = calculateWordProbability(frequenciesDictionary, getWordCount(trainingDataWordList))
    
    return probabilitiesDictionary

def classifyComment(comment: str, posProbabilities: dict, negProbabilities: dict, posWordList: list, negWordList: list) -> str:
    """
    Determines the sentiment (positive or negative) of a given comment.

    Parameters:
    - comment (str): The input comment to be classified.
    - posProbabilities (dict): Dictionary of positive word probabilities.
    - negProbabilities (dict): Dictionary of negative word probabilities.
    - posWordList (list): List of processed positive training words.
    - negWordList (list): List of processed negative training words.

    Returns:
    str: The classified sentiment, which can be "Positive" or "Negative"

    Example:
    >>> classifyComment("Great product!", posProbabilities, negProbabilities, posWordList, negWordList)
    'Positive'
    """
    # Clean the comment to classify
    cleanedCommentList = parseMessage(comment)

    # Calculate the likelyhoods of being positive or negative
    posLikelihood = calculateLikelihood(cleanedCommentList, posProbabilities, posWordList, base_pos_probability)
    negLikelihood = calculateLikelihood(cleanedCommentList, negProbabilities, negWordList, base_neg_probability)

    # Return the result
    return interpretResults(posLikelihood, negLikelihood)

def interpretResults(pos: float, neg: float) -> str:
    """
    Determine the sentiment based on positive and negative scores.
    (Note: It does not find neutral sentiments)

    Parameters:
    - pos (float): The positive score.
    - neg (float): The negative score.

    Returns:
    str: The sentiment classification ("Positive" or "Negative")

    Note:
    This will not determine a neutral sentiment classification.

    Example:
    >>> interpretResults(0.6, 0.4)
    'Positive'
    """

    result = ""

    # Determine which sentiment is more likely (positive or negative)
    if pos >= neg:
        result += "Positive"
    else:
        result += "Negative"

    # Return the sentiment determined
    return result

def calculateLikelihood(commentWordList: list, probabilitiesDict: dict, trainingWordList: list, baseProbability: float) -> float:
    """
    Calculate the likelihood of a comment being positive/negative based on the given parameters.

    Parameters:
    - commentWordList (list): A list of words in the comment.
    - probabilitiesDict (dict): A dictionary of word probabilities.
    - trainingWordList (list): A list of words in the training data.
    - baseProbability (float): The base probability.

    Returns:
    float: The calculated likelihood.

    Example:
    >>> calculateLikelihood(['good', 'great', 'awesome'], {'good': 0.5, 'great': 0.3, 'awesome': 0.2}, ['good', 'bad', 'awesome'], 0.4)
    0.02
    """
    # Initialize likelihood with the base probability
    likelihood = baseProbability

    # Get the number of words in the comment
    totalCommentWords = getWordCount(commentWordList)

    # Calculate word frequencies in the training data
    frequencyOfTrainingWords = calculateFrequencies(trainingWordList)

    # Get the total number of unique words in the training data
    totalTrainingWords = getWordCount(trainingWordList)

    # Counter to track all the words checked
    wordCounter = 0

    # Create a set of training words for faster lookup
    trainingWordSet = set(frequencyOfTrainingWords)

    # Iterate through each word in the comment
    for commentWord in commentWordList:
        # Check if the word is in the training data
        if commentWord in trainingWordSet:
            # If a match is found, multiply the likelihood by the word's probability
            likelihood *= probabilitiesDict.get(commentWord)
            wordCounter += 1

    # Apply smoothing for words not in the training data
    while wordCounter < totalCommentWords:
        likelihood *= 1 / totalTrainingWords
        wordCounter += 1

    return likelihood

def addToTrainingData(comment: str, filePath: str) -> None:
    """
    Append a comment to a training data file, ensuring each new entry starts on a new line.

    Parameters:
    - comment (str): comment (str): A raw comment input by the user to be added to the training data.
    - filePath (str): The path to the file where the comment should be appended.

    Returns:
    None
    """
    # Initialize the variable with a newline character (so that it makes every new entry on a new line)
    commentIntoSingleLine = "\n"

    with open(filePath, "a") as filePath:
        # For all the words in the comment
        for word in comment:
            # Check if the word is not a newline character
            if word != "\n":
                # Concatenate the word to the commentIntoSingleLine
                commentIntoSingleLine += word
            # Check if the word is a newline character
            elif word == "\n":
                # Add a space instead of the newline character
                commentIntoSingleLine += " "

        # Write the commentIntoSingleLine to the file
        filePath.write(commentIntoSingleLine)    

def UpdateClassifierData() -> None:
    """
    Update the global variables used in the sentiment analysis classifier.

    Global Variables Updated:
    - base_pos_probability (float): The base probability rate for positive sentiment.
    - base_neg_probability (float): The base probability rate for negative sentiment.
    - stopwords (list): List of stopwords.
    - posWordList (list): List of words in the positive training data.
    - negWordList (list): List of words in the negative training data.
    - posProbabilities (dict): Dictionary of word probabilities for positive sentiment.
    - negProbabilities (dict): Dictionary of word probabilities for negative sentiment.

    Returns:
    None
    """
    global base_pos_probability
    global base_neg_probability
    global stopwords
    global posWordList
    global negWordList
    global posProbabilities
    global negProbabilities

    # Process Data
    stopwords = convertFileToList(STOPWORDS_DATA)
    posWordList = processData(POS_TRAINING_DATA)
    negWordList = processData(NEG_TRAINING_DATA)

    # Calculate Probabilities
    posProbabilities = calculateSentimentProbabilities(posWordList)
    negProbabilities = calculateSentimentProbabilities(negWordList)

    # Base Probability Rate
    base_pos_probability = 0.5
    base_neg_probability = 0.5

##################################################
#                ASYNC FUNCTIONS                 #
##################################################
async def sendTrainingEmbed(comment: str) -> discord.Embed:
    """
    Send a training embed with a given comment for sentiment classification.

    Parameters:
    - comment (str): The comment to be included in the embed.

    Returns:
    discord.Embed: The sent training embed.

    Example:
    >>> await sendTrainingEmbed("This is a positive comment.")
    <discord.Embed object at 0x...>
    """
    # Creating an embed to prompt the user to select how they would classify their data
    promptEmbed = discord.Embed(title=f"Training",
                                description=f"` {comment} `\n\n{PROMPT_SELECT_SENTIMENT}",
                                color=COLOUR_GREEN)
    promptEmbed.add_field(name=MSG_CLASSIFICATION_OPTIONS, value="", inline=False)

    # Send the embed
    trainingEmbed = await DISCORD_BOT.get_channel(DISCORD_CHANNEL_ID).send(embed=promptEmbed)

    # Add reactions (emojis) to the Embed as options for the user to click
    await trainingEmbed.add_reaction(EMOJI_SMILE)
    await trainingEmbed.add_reaction(EMOJI_FROWN)
    await trainingEmbed.add_reaction(EMOJI_CROSSMARK)

    return trainingEmbed

async def waitForReaction(ctx: discord.ext.commands.Context, embed: discord.Embed):
    """
    Wait for a reaction from the user on the provided embed.

    Parameters:
    - ctx (commands.Context): The context of the command.
    - embed (discord.Embed): The embed to wait for reactions.

    Returns:
    (Reaction or None): The reaction object if a reaction occurs within the timeout,
    or None if no reaction occurs within the specified time.
    """

    # Inner Function: check()
    def check(reaction: discord.Reaction, user: discord.User) -> bool:
        """
        Inner Function: Check if the reaction meets the criteria for the wait_for function.

        Parameters:
        - reaction (discord.Reaction): The reaction object.
        - user (discord.User): The user who added the reaction.

        Returns:
        bool: True if the reaction meets the specified criteria, False otherwise.
        """
        return (
            reaction.count > 1
            and user == ctx.author
            and reaction.message.id == embed.id
            and reaction.emoji in [EMOJI_SMILE, EMOJI_FROWN, EMOJI_CROSSMARK]
        )

    try:
        reaction, user = await DISCORD_BOT.wait_for('reaction_add', timeout=30.0, check=check)
        return reaction
    except asyncio.TimeoutError:
        # If no reaction within the specified time, delete the embed
        await embed.delete()
        return None

async def handlePositiveReaction(ctx: discord.ext.commands.Context, comment: str) -> None:
    """
    Handles a negative reaction by updating training data, updating classifier data with the newly added comment, 
    classifying the comment, and sending results in an embed

    Parameters:
    - ctx (discord.ext.commands.Context): The context of the command.
    - comment (str): The comment to handle.

    Returns:
    None
    """
    # Add the comment to the positive training data
    addToTrainingData(comment, POS_TRAINING_DATA)

    # Refresh data
    UpdateClassifierData()

    # Run the algorithm to determine what the comment was classified as
    classifiersPredictionResult = classifyComment(comment, posProbabilities, negProbabilities, posWordList, negWordList)

    # Send the results in an embed to the user
    await sendClassificationEmbed(ctx, "positive", comment, classifiersPredictionResult)

async def handleNegativeReaction(ctx: discord.ext.commands.Context, comment: str) -> None:
    """
    Handles a negative reaction by updating training data, updating classifier data with the newly added comment, 
    classifying the comment, and sending results in an embed

    Parameters:
    - ctx (discord.ext.commands.Context): The context of the command.
    - comment (str): The comment to handle.

    Returns:
    None
    """
    # Add the comment to the negative training data
    addToTrainingData(comment, NEG_TRAINING_DATA)

    # Refresh data
    UpdateClassifierData()

    # Run the algorithm to determine what the comment was classified as
    classifiersPredictionResult = classifyComment(comment, posProbabilities, negProbabilities, posWordList, negWordList)

    # Send the results in an embed to the user
    await sendClassificationEmbed(ctx, "negative", comment, classifiersPredictionResult)

async def sendClassificationEmbed(ctx: discord.ext.commands.Context, usersClassification: str, comment: str, classifiersPredictionResult: str) -> None:
    """
    Sends an embed displaying the success of adding training data.

    Parameters:
    - ctx (discord.ext.commands.Context): The context of the command.
    - usersClassification (str): The user's classification.
    - comment (str): The comment for which the classification was made.
    - classifiersPredictionResult (str): The predicted classification.

    Returns:
    None
    """
    # Create an Embed displaying their success of adding training data 
    addingDataEmbed = discord.Embed(title=f"{MSG_TITLE_DATA_ADDED}",
                                      description=f"{ctx.author.mention} added to the **{usersClassification}** training data.",
                                      color=COLOUR_GREEN)
    addingDataEmbed.add_field(name="Data:", value=f"` {comment} `", inline=False)
    addingDataEmbed.add_field(name="Your Classification:", value=f"` {usersClassification.capitalize()} `", inline=True)
    addingDataEmbed.add_field(name=f"{DISCORD_BOT.user.name}'s Classification:", value=f"` {classifiersPredictionResult} `", inline=True)
    addingDataEmbed.set_thumbnail(url=f"attachment://{IMG_CHECKMARK}")
    
    # Creating a discord file containing an image
    checkmarkFile = discord.File(f"{PATH_TO_IMAGES+IMG_CHECKMARK}", filename=f"{IMG_CHECKMARK}")

    # Send the Embed
    await DISCORD_BOT.get_channel(DISCORD_CHANNEL_ID).send(file=checkmarkFile, embed=addingDataEmbed)

##################################################
#               DISCORD EVENTS                   #
##################################################
# When the bot turns on
@DISCORD_BOT.event
async def on_ready() -> None:
    """
    Event triggered when the bot has successfully connected to Discord.

    Returns:
    None
    """
    # Console message on bot start-up (For debugging purposes)
    print(f"Connection: Success")
    print(f"BotID: {DISCORD_BOT.user.name}")

    # Update classifier data upon bot start-up
    UpdateClassifierData()

# When a message is sent
@DISCORD_BOT.event
async def on_message(msg: discord.Message) -> None:
    """
    Event triggered when a message is sent.

    Parameters:
    - msg (discord.Message): The message object.

    Returns:
    None
    """
    # Check if the message sender is not the bot itself to prevent an infinite loop
    if msg.author == DISCORD_BOT.user:
        return
    
    # !train - add training data for the bot
    if msg.content.startswith(f"{BOT_PREFIX}train"):
        await DISCORD_BOT.process_commands(msg)

    # !predict - ask the bot to classify a comment input by the user
    if msg.content.startswith(f"{BOT_PREFIX}classify"):
        await DISCORD_BOT.process_commands(msg)

##################################################
#               DISCORD COMMANDS                 #
##################################################
# !train
@DISCORD_BOT.command()
async def train(ctx: discord.ext.commands.Context, *, comment=None) -> None:
    """
    Command to add training data for the bot.

    Parameters:
    - ctx (discord.ext.commands.Context): The context of the command.
    - comment (str): The entire string input by the user in discord following the command !train. This comment will be added to the training data.

    Returns:
    None
    """
    if comment is not None:
        # Send the training embed and wait for a reaction
        trainEmbed = await sendTrainingEmbed(comment)
        reaction = await waitForReaction(ctx, trainEmbed)

        if reaction is not None:
            await trainEmbed.delete()

            # Handle different reactions based on the emoji that was reacted (clicked)
            if reaction.emoji == EMOJI_SMILE:
                await handlePositiveReaction(ctx, comment)
            elif reaction.emoji == EMOJI_FROWN:
                await handleNegativeReaction(ctx, comment)

# !classify
@DISCORD_BOT.command()
async def classify(ctx: discord.ext.commands.Context, *, comment=None) -> None:
    """
    Command to classify a comment using the bot's sentiment analysis.

    Parameters:
    - ctx (discord.ext.commands.Context): The context of the command.
    - comment (str): The entire string input by the user in discord following the command !train. This is the data that will be classified

    Returns:
    None
    """
    if comment is not None:
        # Re-run the learning algorithm to ensure it's reading the most recent data
        UpdateClassifierData()

        # Run the algorithm to determine the prediction result
        predictionResult = classifyComment(comment, posProbabilities, negProbabilities, posWordList, negWordList)

        # Determine color and image based on the prediction
        colorToDisplay = COLOUR_GREEN if predictionResult == "Positive" else COLOUR_RED
        imageToDisplay = IMG_POSITIVE if predictionResult == "Positive" else IMG_NEGATIVE

        # Create an embed displaying the classification results to the user
        classificationEmbed = discord.Embed(
            title=f"Classification",
            description=f"` {predictionResult} `",
            color=colorToDisplay
        )
        classificationEmbed.add_field(name=f"Text:", value=f"` {comment} `", inline=True)
        classificationEmbed.set_thumbnail(url=f"attachment://{imageToDisplay}")

        # Creating a discord file containing an image
        imageFile = discord.File(f"{PATH_TO_IMAGES+imageToDisplay}", filename=f"{imageToDisplay}")

        # Send the embed
        await DISCORD_BOT.get_channel(DISCORD_CHANNEL_ID).send(file=imageFile, embed=classificationEmbed)

##################################################
#            DISCORD BOT START-UP                #  
##################################################
# Start the bot
DISCORD_BOT.run(DISCORD_TOKEN)