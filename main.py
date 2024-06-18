########### Sentiment Analysis ###########

########### Imports ###########
import nltk
import pandas as pd
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def import_sentences_paragraphs(paths_list):
    """
    This function imports the previously found text portions (
    sentences/paragraphs) as returns them as a list of pandas dataframes
    :param paths_list: a list with the paths of the relevant files
    :return: list of pandas dataframes
    """
    dataframes = []
    for path in paths_list:
        df = pd.read_csv(path)
        dataframes.append(df)
    return dataframes


def analyze_sentiment_roberta(text):
    """
    This function analyzes the sentiment found in a text portion (
    sentence/paragraph) using a pre-made version of the RoBERTa model
    :param text: sentence/paragraph
    :return: a list containing the top sentiment for the given text,
    its score, and the full scores list
    """
    # importing the pre-made model
    model_name = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    try:
        # tokenizing the processed text
        encoded_input = tokenizer(text, return_tensors='pt')
        # running inference for sentiment analysis
        # selecting first output as we used only one text as input
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)  # sorting
        ranking = ranking[::-1]
        all_scores = {}
        # condensing all details together
        for i in range(scores.shape[0]):
            label = config.id2label[ranking[i]]
            score = scores[ranking[i]]
            all_scores[label] = score
        sentiment = config.id2label[ranking[0]], scores[ranking[0]], all_scores
    except Exception as e:
        sentiment = Exception("Model related error.")
    return sentiment


def analyze_emotions_roberta(text):
    """
    This function analyzes the emotion found in a text portion (
    sentence/paragraph) using a pre-made version of the RoBERTa model
    :param text: sentence/paragraph
    :return: a list containing the top emotion for the given text,
    its score, and the full scores list
    """
    # importing the pre-made model
    classifier = pipeline(task="text-classification",
                          model="SamLowe/roberta-base-go_emotions", top_k=None)
    model_outputs = classifier(text)
    # all_labels_scores is of the form:
    # [{'label': 'neutral', 'score': 0.8592366576194763},
    # {'label': 'approval', 'score': 0.06880379468202591}...
    all_labels_scores = model_outputs[0]
    top_emotion = all_labels_scores[0]['label']
    top_emotion_score = round(all_labels_scores[0]['score'], 4)
    return [top_emotion, top_emotion_score, all_labels_scores]


def analyze_emotions_dis_roberta(text):
    """
    This function analyzes the emotion found in a text portion (
    sentence/paragraph) using a pre-made version of the DistilRoBERTa model
    :param text: sentence/paragraph
    :return: a list containing the top emotion for the given text,
    its score, and the full scores list
    """
    # importing the pre-made model
    classifier = pipeline("text-classification",
                          model="j-hartmann/emotion-english-distilroberta-base",
                          return_all_scores=True)
    scores = classifier(text)[0]
    # sorting the scores
    sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)
    top_emotion = sorted_scores[0]['label']
    top_emotion_score = round(sorted_scores[0]['score'], 4)
    # printing results for follow up while running
    print([top_emotion, top_emotion_score, sorted_scores])
    return [top_emotion, top_emotion_score, sorted_scores]


def analyze_sentiment_vader(text):
    """
    This function analyzes the sentiment found in a text portion (
    sentence/paragraph) using a pre-made version of the VADER model
    :param text: sentence/paragraph
    :return: a list containing the top sentiment for the given text,
    its score, and the full scores list
    """
    # abbreviation to full name dictionary for later use
    sentiment_mapping = {'neg': 'negative', 'neu': 'neutral',
                         'pos': 'positive', 'compound': 'compound'}
    # importing the model
    analyzer = SentimentIntensityAnalyzer()
    results_dict = analyzer.polarity_scores(text)
    mapped_results_dict = \
        {sentiment_mapping[k]: v for k, v in results_dict.items()}
    # a version of the dictionary without the compound score
    dict_wo_compound = dict(list(mapped_results_dict.items())[:3])
    sorted_dict = dict(sorted(dict_wo_compound.items(),
                              key=lambda item: item[1],
                              reverse=True))
    sentiment = list(sorted_dict.keys())[0]
    sentiment_score = list(sorted_dict.values())[0]
    # printing results for follow up while running
    print([sentiment, sentiment_score, mapped_results_dict])
    return [sentiment, sentiment_score, mapped_results_dict]


def run_sentiment_analysis(df, text_col, model):
    """
    This function executes sentiment/emotion analysis on one of the
    dataframes (sentences/paragraphs) using the given model name
    :param df: the dataframe containing the text samples (sentences/paragraphs)
    :param text_col: a marker of whether the text is sentences/paragraphs
    :param model: which model to use
    :return: nothing, saves the results
    """
    if model == "roberta":
        # running relevant sentiment RoBERTa function and storing the results
        # as a new dataframe
        df[['rob_sentiment', 'rob_score', 'rob_all_scores']] = (
            df[text_col].apply(
                lambda x: pd.Series(analyze_sentiment_roberta(x))))
        # saving the dataframe as a file
        df.to_csv(f"sent_analyzed_{text_col}s.csv", index=False)
    if model == "roberta-emotions":
        # running relevant emotion RoBERTa function and storing the results
        # as a new dataframe
        df[['rob_emotion', 'rob_emotion_score', 'all_emotion_scores']] = (
            df[text_col].apply(
                lambda x: pd.Series(analyze_emotions_roberta(x))))
        # saving the dataframe as a file
        df.to_csv(f"rob_emotion_analyzed_{text_col}s.csv", index=False)
    if model == "dis-roberta-emotions":
        # running relevant emotion DistilRoBERTa function and storing the
        # results as a new dataframe
        df[['dis_rob_emotion', 'dis_rob_emotion_score',
            'dis_all_emotion_scores']] = (
            df[text_col].apply(
                lambda x: pd.Series(analyze_emotions_dis_roberta(x))))
        # saving the dataframe as a file
        df.to_csv(f"dis_rob_emotion_analyzed_{text_col}s.csv", index=False)
    if model == "vader":
        # running relevant sentiment VADER function and storing the results
        # as a new dataframe
        df[['v_sentiment', 'v_sentiment_score',
            'v_all_scores']] = (
            df[text_col].apply(
                lambda x: pd.Series(analyze_sentiment_vader(x))))
        # saving the dataframe as a file
        df.to_csv(f"vader_sentiment_{text_col}s.csv", index=False)
    print(df)


if __name__ == "__main__":
    ########### RoBERTa sentiment analysis ###########
    # files_paths = ['lem_book_sentences.csv',
    #                'lem_book_paragraphs.csv',
    #                'lem_book_test.csv']
    # df_list = import_sentences_paragraphs(files_paths)
    # sentences_df, paragraphs_df, test_df = df_list[0], df_list[1], df_list[2]
    # run_sentiment_analysis(sentences_df, "sentence", "roberta")
    # run_sentiment_analysis(paragraphs_df, "paragraph", "roberta")
    ########### RoBERTa emotion analysis ###########
    # sent_analyzed_paths = ['sent_analyzed_sentences.csv',
    #                        'sent_analyzed_paragraphs.csv']
    # sent_df_list = import_sentences_paragraphs(sent_analyzed_paths)
    # sent_sentences_df, sent_paragraphs_df = sent_df_list[0], sent_df_list[1]
    # run_sentiment_analysis(sent_sentences_df, "sentence",
    #                        "roberta-emotions")
    # run_sentiment_analysis(sent_paragraphs_df, "paragraph",
    #                        "roberta-emotions")
    ########### DistilRoBERTa emotion analysis ###########
    # emot_analyzed_paths = ['rob_emotion_analyzed_sentences.csv',
    #                        'rob_emotion_analyzed_paragraphs.csv']
    # emot_df_list = import_sentences_paragraphs(emot_analyzed_paths)
    # emot_sentences, emot_paragraphs = emot_df_list[0], emot_df_list[1]
    # run_sentiment_analysis(emot_sentences, "sentence",
    #                        "dis-roberta-emotions")
    # run_sentiment_analysis(emot_paragraphs, "paragraph",
    #                        "dis-roberta-emotions")
    ########### VADER sentiment analysis ###########
    spell_col_paths = ['sentences_w_spell_col.csv',
                       'paragraphs_w_spell_col.csv']
    spell_col_dfs = import_sentences_paragraphs(spell_col_paths)
    spell_sentences, spell_paragraphs = spell_col_dfs[0], spell_col_dfs[1]
    run_sentiment_analysis(spell_sentences, 'sentence', 'vader')
    run_sentiment_analysis(spell_paragraphs, 'paragraph', 'vader')
