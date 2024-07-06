########### Imports ###########
from main import import_sentences_paragraphs
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


########### Global variables ###########
MODELS_DICT = {'rob_sentiment': 'RoBERTa', 'v_sentiment': 'VADER',
               'rob_emotion': 'RoBERTa', 'dis_rob_emotion': 'DistilRoBERTa'}


def sentiment_across_books(df, model_col_name, text_col):
    """
    This function creates a plot showing the presence of different sentiments
    throughout the different books
    :param df: the dataframe containing the text samples (sentences/paragraphs)
    :param model_col_name: which model sentiment analysis to use
    (RoBERTa/VADER)
    :param text_col: a marker of whether the text is sentences/paragraphs
    :return: nothing, generates a plot
    """
    # creating a new column 'book_num_adj' where 1 is added to each book
    # number since the original counting starts from 0
    df['book_num_adj'] = df['book_num'] + 1
    plt.figure(figsize=(12, 6))  # setting plot size
    # creating a count plot
    ax = sns.countplot(data=df, x='book_num_adj', hue=model_col_name)
    # adding annotations on top of each bar
    for p in ax.patches:
        height = int(p.get_height())
        if height > 0:
            ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 5),
                        textcoords='offset points')
    # setting the title and labels
    plt.title(f'Distribution of {MODELS_DICT[model_col_name]} {text_col}s\' '
              f'Sentiments Across Books')
    plt.xlabel('Book Number')
    plt.ylabel('Sentiment Counts')
    plt.show()


def count_emotions_by_book(df, model_col_name):
    """
    This function creates a dictionary counting the appearances of each
    emotion detected throughput the books
    :param df: the dataframe containing the text samples (sentences/paragraphs)
    :param model_col_name: which model emotion analysis to use
    (RoBERTa/DistilRoBERTa)
    :return:the dictionary created
    """
    emotion_counts = {}
    # iterating through the dataFrame and updating the counts
    for _, row in df.iterrows():
        book_num = str(int(row['book_num']) + 1)
        emotion = row[model_col_name]
        try:
            emotion_counts[f"book {book_num}: {emotion}"] += 1
        except KeyError:
            emotion_counts[f"book {book_num}: {emotion}"] = 1
    # emotion counts is of the form:
    # {'book 1: neutral': 9, 'book 2: neutral': 10, 'book 2: annoyance': 1...
    return emotion_counts


def emotion_across_books(df, model_col_name, text_col):
    """
    This function creates a plot showing the presence of different emotions
    throughout the different books
    :param df: the dataframe containing the text samples (sentences/paragraphs)
    :param model_col_name: which model emotions analysis to use
    (RoBERTa/DistilRoBERTa)
    :param text_col: a marker of whether the text is sentences/paragraphs
    :return: nothing, generates a plot
    """
    plot_data = count_emotions_by_book(df, model_col_name)
    # converting the relevant  data to a new dataframe for easier plotting with
    # Seaborn
    df = pd.DataFrame(list(plot_data.items()),
                      columns=['Book and Emotion', 'Count'])
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x='Book and Emotion', y='Count',
                     palette='rocket_r', hue='Book and Emotion', legend=False)
    # adding counts above the bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center', xytext=(0, 5),
                    textcoords='offset points')
    # setting titles and labels
    plt.title(f'Distribution of {MODELS_DICT[model_col_name]} {text_col}s\' '
              f'Emotions Across Books')
    plt.xlabel('Book and Emotion')
    plt.ylabel('Count of Emotions')
    plt.xticks(rotation=45, ha='right',
               fontsize=8)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()


def count_spells_by_sentiment(df, model_col_name):
    """
    This function counts spells appearances for each sentiment found (
    negative, positive and neutral)
    :param df: the dataframe containing the text samples (sentences/paragraphs)
    :param model_col_name: which model sentiment analysis to use
    (RoBERTa/VADER)
    :return: 3 dictionaries, dict per sentiment
    """
    negative_dict, positive_dict, neutral_dict = {}, {}, {}
    # mapping sentiment names to actual dictionaries
    sentiment_to_dict = {'negative': negative_dict, 'positive': positive_dict,
        'neutral': neutral_dict}
    # iterating through the DataFrame and update the counts
    for _, row in df.iterrows():
        sentiment = row[model_col_name]
        spell = row['spell']
        sentiment_dict = sentiment_to_dict[sentiment]
        try:
            sentiment_dict[spell] += 1
        except KeyError:
            sentiment_dict[spell] = 1
    # sorting the 3 dictionaries by appearances
    sorted_negative = dict(
        sorted(negative_dict.items(), key=lambda item: item[1], reverse=True))
    sorted_positive = dict(
        sorted(positive_dict.items(), key=lambda item: item[1], reverse=True))
    sorted_neutral = dict(
        sorted(neutral_dict.items(), key=lambda item: item[1], reverse=True))
    return [sorted_negative, sorted_positive, sorted_neutral]


def top_spell_by_sentiment(df, model_col_name):
    """
    This function plots the top 10 spells (if there are) found for each
    sentiment classification.
    :param df: the dataframe containing the text samples (sentences/paragraphs)
    :param model_col_name: which model sentiment analysis to use
    (RoBERTa/VADER)
    :return: nothing, generates the plots
    """
    sentiments_list = ['Negative', 'Positive', 'Neutral']
    # generating the appearances dictionaries
    dicts_list = count_spells_by_sentiment(df, model_col_name)
    for i in range(len(dicts_list)):
        print(dicts_list[i])  # for follow up
        if len(dicts_list[i]) > 0:
            # slicing the top 10 spells
            top_spells = list(dicts_list[i].items())[:10]
            data = [{'Spell': spell, 'Count': count,
                     'Sentiment': sentiments_list[i]} for
                    spell, count in top_spells]
            df = pd.DataFrame(data)  # converting the data to a new dataframe
            plt.figure(figsize=(14, 8))
            ax = sns.barplot(data=df, x='Spell', y='Count', palette='viridis',
                             hue='Spell')
            # setting titles and labels
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{int(height)}',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='center', xytext=(0, 5),
                            textcoords='offset points')
            plt.title(f'Top 10 Spells Classified as '
                      f'{sentiments_list[i]} By {MODELS_DICT[model_col_name]}')
            plt.xlabel('Spell')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()


def count_spells_by_emotion(df, model_col_name, emotions_list):
    """
    This function counts spells appearances for each emotion found (
    list is different between the 2 models and passed as a parameter)
    :param df: the dataframe containing the text samples (sentences/paragraphs)
    :param model_col_name: which model emotions analysis to use
    (RoBERTa/DistilRoBERTa)
    :param emotions_list: a list of all possible emotions possible in the model
    :return: a dictionary of nested dictionaries (one per emotion)
    """
    # creating the initial high-level dictionary
    emotions_dicts = {emotion: {} for emotion in emotions_list}
    # iterating through the dataframe and updating the counts
    for _, row in df.iterrows():
        emotion = row[model_col_name]
        spell = row['spell']
        if spell in emotions_dicts[emotion].keys():
            emotions_dicts[emotion][spell] += 1
        elif spell not in emotions_dicts[emotion].keys():
            emotions_dicts[emotion][spell] = 1
    # sorting the generated dictionaries
    sorted_emotions_dicts = {}
    for emotion, emot_spells in emotions_dicts.items():
        sorted_emot_spells = dict(
        sorted(emot_spells.items(), key=lambda item: item[1], reverse=True))
        if len(sorted_emot_spells) > 0:
            sorted_emotions_dicts[emotion] = sorted_emot_spells
    return sorted_emotions_dicts


def top_spell_by_emotion(df, model_col_name):
    """
    This function plots the top 10 spells (if there are) found for each
    emotion classification.
    :param df: the dataframe containing the text samples (sentences/paragraphs)
    :param model_col_name: which model emotions analysis to use
    (RoBERTa/DistilRoBERTa)
    :return: nothing, generates the plots
    """
    emotions_list = []
    # initializing a list of all possible emotions in each model for later use
    if model_col_name == 'rob_emotion':
        emotions_list = ['admiration', 'amusement', 'anger', 'annoyance',
                         'approval', 'caring', 'confusion', 'curiosity',
                         'desire', 'disappointment', 'disapproval',
                         'disgust', 'embarrassment', 'excitement', 'fear',
                         'gratitude', 'grief', 'joy', 'love', 'nervousness',
                         'optimism', 'pride', 'realization', 'relief',
                         'remorse', 'sadness', 'surprise', 'neutral']
    elif model_col_name == 'dis_rob_emotion':
        emotions_list = ['anger', 'disgust', 'fear', 'joy', 'neutral',
                         'sadness', 'surprise']
    # generating the appearances dictionaries
    emotion_dicts = count_spells_by_emotion(df, model_col_name, emotions_list)
    # generating the plots
    for emotion, emotion_dict in emotion_dicts.items():
        print(emotion, emotion_dict)  # for follow up
        # slicing the top 10 spells
        top_spells = list(emotion_dict.items())[:10]
        data = [{'Spell': spell, 'Count': count,
                 'Emotion': emotion} for
                spell, count in top_spells]
        df = pd.DataFrame(data)  # converting the data to a new dataframe
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(data=df, x='Spell', y='Count', palette='viridis',
                         hue='Spell')
        # setting titles and labels
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{int(height)}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 5),
                        textcoords='offset points')
        plt.title(f'Top 10 Spells Classified as '
                  f'{emotion} By {MODELS_DICT[model_col_name]}')
        plt.xlabel('Spell')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


########### Importing the data  ###########
files_paths = ['vader_sentiment_sentences.csv',
               'vader_sentiment_paragraphs.csv']
df_list = import_sentences_paragraphs(files_paths)
sentences_df, paragraphs_df = df_list[0], df_list[1]
########### Distribution of sentiment by books ###########
# sentiment_across_books(sentences_df, 'rob_sentiment', 'Sentence')
# sentiment_across_books(sentences_df, 'v_sentiment', 'Sentence')
# sentiment_across_books(paragraphs_df, 'rob_sentiment', 'Paragraph')
# sentiment_across_books(paragraphs_df, 'v_sentiment', 'Paragraph')
########### Distribution of emotion by books ###########
# emotion_across_books(sentences_df, 'rob_emotion', 'Sentence')
# emotion_across_books(sentences_df, 'dis_rob_emotion', 'Sentence')
# emotion_across_books(paragraphs_df, 'rob_emotion', 'Paragraph')
# emotion_across_books(paragraphs_df, 'dis_rob_emotion', 'Paragraph')
########### Top spells for sentiment ###########
# top_spell_by_sentiment(paragraphs_df, 'rob_sentiment')
# top_spell_by_sentiment(paragraphs_df, 'v_sentiment')
########### Top spells for emotion ###########
# top_spell_by_emotion(paragraphs_df, 'rob_emotion')
# top_spell_by_emotion(paragraphs_df, 'dis_rob_emotion')
