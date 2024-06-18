########### Imports ###########
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import spacy
import pandas as pd
import ast


########### Global variables ###########
BOOK_DICT = {"Philosopher's Stone": "books_txt/Harry-Potter-1_-Rowling_"
                "-J-K-Harry-Potter-and-the-Philosopher_s-Stone.txt",
              "Chamber of Secrets": "books_txt/Harry-Potter-2_-Rowling_"
                "-J-K-Harry-Potter-and-the-Chamber-of-Secrets.txt",
              "Prisoner of Azkaban": "books_txt/Harry-Potter-3_-Rowling_"
                "-J-K-Harry-Potter-and-the-Prisoner-of-Azkaban.txt",
              "Goblet of Fire": "books_txt/Harry-Potter-4_-Rowling_"
                "-J-K-Harry-Potter-and-the-Goblet-of-Fire.txt",
              "Order of the Phoenix": "books_txt/Harry-Potter-5_-Rowling_"
                "-J-K-Harry-Potter-and-the-Order-of-the-Phoenix.txt",
              "Half-Blood Prince": "books_txt/Harry-Potter-6_-Rowling_"
                "-J-K-Harry-Potter-and-the-Half-Blood-Prince.txt",
              "Deathly Hallows": "books_txt/Harry-Potter-7_-Rowling_"
                "-Joanne-Kathleen_-GrandPr__-Mary-_illustrations_"
                                 "-Harry-Potter-and-the-Deathl.txt"}

########### Global variables - manual sentence editing ###########
PACK_PARAGRAPH = ("But we’ve got to get going, Harry, we’re supposed to be "
                  "packing,\" she added guiltily, looking around at all the "
                  "mess on the floor. \"Oh — yeah,\" said Harry, grabbing up a "
                  "few more books. \"Don’t be stupid, it’ll be much quicker "
                  "if I — pack!\" cried Tonks, waving her wand in a long, "
                  "sweeping movement over the floor. Books, clothes, "
                  "telescope, and scales all soared into the air and flew "
                  "pell-mell into the trunk. \"It’s not very neat,\" said "
                  "Tonks, walking over to the trunk and looking down at the "
                  "jumble inside. ")

POINT_PARAGRAPH_1 = ("The maze was growing darker with every passing minute as "
                     "the sky overhead deepened to navy. He reached a second "
                     "fork. \"Point Me,\" he whispered to "
                     "his wand, holding it flat in his palm. The wand spun "
                     "around once and pointed toward his right, into solid "
                     "hedge. That way was north, and he knew that he needed to "
                     "go northwest for the center of the maze.")

POINT_PARAGRAPH_2 = ("Harry broke into a run. He had a choice of paths up ahead"
                     ".\"Point Me!\" he whispered again to his wand, and it "
                     "spun around and pointed him to the right-hand one. He "
                     "dashed up this one and saw light ahead. The Triwizard "
                     "Cup was gleaming on a plinth a hundred yards away.")


def import_spells():
    """
    This function import the previously scraped list of spells that was saved
    as a txt file.
    :return: the file contents as a list
    """
    with open("hp_spells_list.txt", 'r') as file:
        content = file.readlines()
        # Remove newline characters from each line and strip whitespace
        spells_list = [line.strip() for line in content]
    return spells_list


def import_books():
    """
    This function imports the content of all Harry Potter books (saved as txt
    files).
    :return: A list of lists, containing all sentences within all books
    """
    books_list = []
    # list of lists (one per book)
    for book, path in BOOK_DICT.items():
        with open(path, 'r', encoding='utf-8') as file:
            # list of all sentences within a book
            book_content = file.readlines()
            book_content = [line for line in book_content if line != '\n']
        books_list.append(book_content)
    return books_list


def filter_punctuation_stopwords(tokens_list):
    """
    This function filters out punctuation characters and stopwords from the
    books sentences.
    :param tokens_list: a list of the sentence's tokens
    :return: the filtered tokens
    """
    # removing punctuation
    filtered_tokens = [re.sub(r'[^\w\s]', '', token)
                       for token in tokens_list]
    # removing stopwords & empty strings
    filtered_tokens = [token.lower() for token in filtered_tokens if token
                       not in stopwords.words('english')
                       and len(token) > 0]
    return filtered_tokens


def tokenize_sentences(books_text):
    """
    This function tokenizes all sentences in each book using nltk's
    sen_tokenize.
    :param books_text: the overall contents of all books
    :return: the books' contents as a tokenized list
    """
    tokenized_books = []
    for book in books_text:  # book_text is alist of lists
        book_tokenized_sentences = []
        for line in book:
            sentences = sent_tokenize(line)  # sentence tokenization
            book_tokenized_sentences += sentences
        tokenized_books.append(book_tokenized_sentences)
    return tokenized_books


def tokenize_words(books_text):
    """
    This function tokenizes each sentence in each book using nltk's
    word_tokenize.
    :param books_text: the overall contents of all books
    :return: the books' contents as a tokenized sentences
    """
    tokenized_books = []
    for book in books_text:  # book_text is alist of lists
        book_tokenized_words = []
        for sentence in book:
            tokens = word_tokenize(sentence)  # word tokenization
            # removing punctuation and stopwords
            tokens = filter_punctuation_stopwords(tokens)
            book_tokenized_words.append(tokens)
        tokenized_books.append(book_tokenized_words)
    return tokenized_books


def find_spells_in_books(book_list, spells_list):
    """
    This function finds all sentences in the books which contain one or more of
    the  spells from the list. It saves the sentences themselves into one
    list, and a slightly bigger sample (the sentence with additional 2
    sentences before and after) which will be referred as 'paragraph'.
    :param book_list: the books' contents after going through both sentence
    and word tokenization
    :param spells_list: the previously scraped and saved spells list
    :return: a general "finished" message. saves all the results as csv files.
    """
    all_books_spells_sentences = []
    all_books_spells_paragraphs = []
    for i in range(len(book_list)):  # book_list[i] == book
        book_spells_sentences = []
        book_spells_paragraphs = []
        book_length = len(book_list)
        for j in range(len(book_list[i])):  # book_list[i][j] == sentence
            # general progress message
            print(f"working on book {i}, sentence {j}/{book_length}")
            # checking for presence of a spell in the sentence
            intersection = set(book_list[i][j]) & set(spells_list)
            if len(intersection) > 0:  # if at least one spell appears
                # adding the sentence itself
                book_spells_sentences.append(book_list[i][j])
                # adding the sentence and the surrounding section
                paragraph = book_list[i][j-2] + book_list[i][j-1] + book_list[
                    i][j] + book_list[i][j+1] + book_list[i][j+2]
                book_spells_paragraphs.append(paragraph)
        all_books_spells_sentences.append(book_spells_sentences)
        all_books_spells_paragraphs.append(book_spells_paragraphs)
    # flattening the nested lists and creating lists of tuples
    flattened_data_sentences = [(book_num, sentence)
                      for book_num, sentences in
                      enumerate(all_books_spells_sentences)
                      for sentence in sentences]
    flattened_data_paragraphs = [(book_num, paragraph)
                      for book_num, paragraphs in
                      enumerate(all_books_spells_paragraphs)
                      for paragraph in paragraphs]
    # converting to a pandas dataframe
    df_sentences = pd.DataFrame(flattened_data_sentences, columns=['book_num',
                                                         'sentence'])
    df_paragraphs = pd.DataFrame(flattened_data_paragraphs, columns=['book_num',
                                                         'paragraph'])
    # saving the dataframe as CSV files
    csv_file_path_sentences = 'all_books_spells_sentences.csv'
    csv_file_path_paragraphs = 'all_books_spells_paragraphs.csv'
    df_sentences.to_csv(csv_file_path_sentences, index=False)
    df_paragraphs.to_csv(csv_file_path_paragraphs, index=False)
    return "Finished"

# 2 spells from the overall list were problematic as they contained regular
# words that appeared in other contexts as well ("pack" and "point me"). I
# decided to remove them from the list, run the overall tracing, and later add
# the specific sentences in which they appear manually (location mentioned
# explicitly in several fan websites). After adding them I still had to
# process and filter them, so I created the following tailored function for
# those few sentences.


def pack_point_sentences(sentences_list):
    """
    This function filters and procceses the specific sentences with the
    spells "pack" and "point me".
    :param sentences_list: a list of the relevant sentences
    :return: the sentences after tokenization
    """
    # add list layer to imitate book structure
    demi_book = [sentences_list]
    tokenized_sentences = tokenize_sentences(demi_book)
    tokenized_words = tokenize_words(tokenized_sentences)
    return tokenized_words


def import_sentences_paragraphs(paths_list, ast_marker=True):
    """
    After the previous functions save the results into CSV files this
    function imports those CSV files as pandas dataframes for the next phases.
    :param paths_list: the paths of the relevant files
    :param ast_marker: marker for whether a specific column has to converted
    back to a list or can remain as a string (all contents are imported as a
    string by default).
    :return: a list with all dataframes created
    """
    dataframes = []
    for path in paths_list:
        df = pd.read_csv(path)
        if ast_marker:
            # Convert the string representation of lists back to actual lists
            df.iloc[:, 1] = df.iloc[:, 1].apply(ast.literal_eval)
        dataframes.append(df)
    return dataframes


def lemmatize_line(line):
    """
    This function receives a portion of text (sentence or paragraph and
    returns only the lemmas of the words in it using spacy's lemmatizer.
    :param line: sentence or paragraph
    :return: a text portion of lemmas only
    """
    nlp = spacy.load('en_core_web_sm')  # initializing the lemmatizer
    doc = nlp(' '.join(line))
    # general message for following the code's progress
    print(' '.join([token.lemma_ for token in doc]))
    return ' '.join([token.lemma_ for token in doc])


def run_lemmatization(df, text_type):
    """
    This function takes the previously saved dataframes and applies
    lemmatization to the found sentences/paragraphs within them.
    :param df: the processed dataframes
    :param text_type: sentences/paragraphs
    :return: the lemmatized dataframe
    """
    # text type - either 'sentence' or 'paragraph'
    # applying the lemmataizing function
    df[text_type] = df[text_type].apply(lemmatize_line)
    df.to_csv(f"lem_book_{text_type}s.csv", index=False)
    return df


def find_spells_in_text(text, spells_list):
    """
    This function traces the spell found in a text portion
    (sentences/paragraphs)
    :param text: sentence/paragraph
    :param spells_list: the full spell list
    :return: the intersection between the spells list and the text portion
    """
    intersection = set(text.split(" ")) & set(spells_list)
    return " ".join(intersection)


def create_spell_column(df, spells_list, text_col):
    """
    For later plots, this function creates a column called 'spell' near each
    sentence/paragraph that contain the spell/s the appear in it.
    :param df: the previously generated dataframes
    :param spells_list: the full spell list
    :param text_col: sentence/paragraph
    :return: nothing, saves the new dataframe
    """
    spell_col = df[text_col].apply(
        lambda x: pd.Series(find_spells_in_text(x, spells_list)))
    df.insert(2, 'spell', spell_col)
    df.to_csv(f"{text_col}s_w_spell_col.csv", index=False)
    print(df.columns)


########### Importing the data ###########
spells_list = import_spells()  # 194 spells total
# lower all caps and split more than 1-word spells
spells_list = [word.lower() for spell in spells_list for word in spell.split()]
# raw_books = import_books()
# # returns a list: [["Harry Potter and the Sorcerer's Stone\n", 'CHAPTER
# ONE\n',...
########### Preparing the data ###########
# tokenized_sentences_data = tokenize_sentences(raw_books)
# # returns a list: [["Harry Potter and the Sorcerer's Stone",
# # 'CHAPTER ONE', 'THE BOY WHO LIVED'...
# tokenized_words_data = tokenize_words(tokenized_sentences_data)
# # returns a list: [[['harry', 'potter', 'sorcerer', 'stone'],
# # ['chapter', 'one'], ['the', 'boy', 'who', 'lived'],
########### Finding all sentences with spells ###########
# all_spells_sentences = find_spells_in_books(tokenized_words_data, spells_list)
# print(all_spells_sentences)
########### Pack-point sentences section ###########
# pack_point_list = [PACK_PARAGRAPH, POINT_PARAGRAPH_1, POINT_PARAGRAPH_2]
# print(pack_point_sentences(pack_point_list))
########### Lemmas section ###########
# files_paths = ['all_books_spells_sentences.csv',
#                'all_books_spells_paragraphs.csv']
# df_list = import_sentences_paragraphs(files_paths)
# print(run_lemmatization(df_list[0], "sentence"))
# print(run_lemmatization(df_list[1], "paragraph"))
########### Add spells column section ###########
files_paths = ['dis_rob_emotion_analyzed_sentences.csv',
               'dis_rob_emotion_analyzed_paragraphs.csv']
df_list = import_sentences_paragraphs(files_paths, False)
df_sentences, df_paragraphs = df_list[0], df_list[1]
create_spell_column(df_sentences, spells_list, "sentence")
create_spell_column(df_paragraphs, spells_list, "paragraph")
