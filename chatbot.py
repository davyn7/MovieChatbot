import util
import porter_stemmer
import numpy as np


# noinspection PyMethodMayBeStatic
class Chatbot:
    
    def __init__(self, creative=False):
        self.name = 'moviebot'
        self.creative = creative

        """ 
        This matrix has the following shape: num_movies x num_users.
        The values stored in each row i and column j is the rating for movie i by user j.
        """
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = ratings
        self.ratings = self.binarize(self.ratings)

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        greeting_message = "Hello! I would like to recommend you some movies! Please tell me about a movie that you have watched."
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        goodbye_message = "Have a nice day!"
        return goodbye_message

    recommending = False
    binary_clarifying = (False, 0, 0)
    multiple_clarifying = (False, 0, [])
    rec_count = 1
    user_movies = {}
    movies = util.load_titles("data/movies.txt")
    affirm = ["yeah", "yes", "ya", "sure", "of course", "ok"]
    negate = ["no", "no, thank you", "no thanks", "nah", "nope"]
    punctuations = ['.', ',', '?', '!', ':', ';']

    def binary_clarification_follow_up(self, line):
        if line in self.affirm:
            response = self.echo_sentiment(self.binary_clarifying[2], self.binary_clarifying[1])
            self.binary_clarifying = (False, 0, 0)
        elif line in self.negate:
            response = "I'm sorry. Please tell me about another movie then."
            self.binary_clarifying = (False, 0, 0)
        else:
            response = "I'm sorry, I do not understand. Please answer yes or no."
        return response

    def multiple_clarification_follow_up(self, line):
        if line in self.negate:
            response = "I'm sorry. Please tell me about another movie then."
            self.multiple_clarifying = (False, 0, [])
        candidates = self.disambiguate(line, self.multiple_clarifying[2])
        val = self.multiple_clarifying[1]
        if len(candidates) == 0:
            response = "I'm sorry, please be more specific. Which movie did you mean?"
        elif len(candidates) == 1:
            response = self.echo_sentiment(candidates[0], val)
            self.multiple_clarifying = (False, 0, [])
        else:
            response = self.multiple_matches(candidates, val)
            self.multiple_clarifying = (True, val, candidates)
        return response

    def invalid_input(self, line):
        sentiment = self.extract_sentiment(line)
        if sentiment > 0:
            response = "I am glad you are happy! If you would like, please tell me about a movie that you watched."
        elif sentiment < 0:
            response = "Oh no, I am sorry if I made you feel bad."
        else:
            response = "I'm sorry, I don't understand what you said. Let's go back to movies!"
        return response

    def no_such_movie(self, movie, sentiment):
        possible_movies = self.find_movies_closest_to_title(movie)
        if len(possible_movies) == 0:
            response = "I'm sorry, I have not seen that movie. Could you tell me about another movie?"
        else:
            response = "Did you mean {}".format(self.movies[possible_movies[0]][0])
            for i in range(1, len(possible_movies)):
                response += " or {}".format(self.movies[possible_movies[i]][0])
            response += "?"
            if len(possible_movies) == 1:
                self.binary_clarifying = (True, sentiment, possible_movies[0])
            else:
                self.multiple_clarifying = (True, sentiment, possible_movies)
        return response

    def multiple_query(self, tuple_list):
        good_list = []
        bad_list = []
        response = ''
        for pair in tuple_list:
            titles = self.find_movies_by_title(pair[0])
            if len(titles) == 1:
                good_list.append((titles[0], pair[1]))
            else:
                bad_list.append(pair)
        if len(good_list) > 0:
            if len(good_list) == 1:
                response = self.echo_sentiment(good_list[0][0], good_list[0][1])
            else:
                response = self.echo_sentiment_multiple(good_list)
        if len(bad_list) > 0:
            for problem in bad_list:
                movie_id = self.find_movies_by_title(problem[0])
                if len(movie_id) == 0:
                    response += ' ' + self.no_such_movie(problem[0], problem[1])
                else:
                    response += ' ' + self.multiple_matches(movie_id)
        return response

    def multiple_matches(self, id_list, sentiment):
        response = "I found {} movies that match your search! Could you specify which one among the list is the movie you watched? I found: ".format(
            len(id_list))
        for i in range(len(id_list) - 1):
            response += "{}, ".format(self.movies[id_list[i]][0])
        response += "and {}.".format(self.movies[id_list[-1]][0])
        self.multiple_clarifying = (True, sentiment, id_list)
        return response

        def correction(self, index, sentiment):
            if self.user_movies[index] == sentiment:
                response = "You have already told me about this movie! Please tell me about another one."
            else:
                if sentiment > 0:
                    response = "I'm sorry. You liked {}. Please tell me about another movie!".format(
                        self.movies[index][0])
                elif sentiment < 0:
                    response = "I'm sorry. You didn't like {}. Please tell me about another movie!".format(
                        self.movies[index][0])
                else:
                    response = "I'm sorry, I'm not sure whether you liked {}. Could you please tell me whether you liked it?".format(
                        self.movies[index][0])
                self.user_movies[index] = sentiment

        return response

    def echo_sentiment(self, index, sentiment):
        if sentiment == 0:
            response = "I'm sorry, I'm not sure whether you liked {}. Could you please tell me whether you liked it?".format(
                self.movies[index][0])
        else:
            if sentiment > 0:
                line_0 = "I see, you liked {}. Cool! Can you tell me about another movie you watched?".format(
                    self.movies[index][0])
                line_1 = "Ok, {} was to your liking. Let's hear about another movie!".format(self.movies[index][0])
                line_2 = "I'm glad you liked {}. Would you tell me about a different movie?".format(
                    self.movies[index][0])
                lines = [line_0, line_1, line_2]
                num = np.random.randint(0, 2)
                response = lines[num]

            else:
                line_0 = "I see, you didn't like {}. I'm sorry. Can you tell me about another movie you watched?".format(
                    self.movies[index][0])
                line_1 = "Ah, it's unfortunate that you didn't like {}. Let's talk about another movie!".format(
                    self.movies[index][0])
                line_2 = "It sucks that you had to watch {}. Let's move on and talk about another movie.".format(
                    self.movies[index][0])
                lines = [line_0, line_1, line_2]
                num = np.random.randint(0, 2)
                response = lines[num]
            self.user_movies[index] = sentiment
        return response

    def echo_sentiment_multiple(self, movie_sentiment):
        unsure = False
        if movie_sentiment[0][1] > 0:
            response = "Ok, you liked {} and ".format(self.movies[movie_sentiment[0][0]][0])
            self.user_movies[movie_sentiment[0][0]] = movie_sentiment[0][1]
        elif movie_sentiment[0][1] == 0:
            response = "Sorry, I am not sure whether you liked {} and ".format(self.movies[movie_sentiment[0][0]][0])
            unsure = True
        else:
            response = "Alright, you did not like {} and ".format(self.movies[movie_sentiment[0][0]][0])
            self.user_movies[movie_sentiment[0][0]] = movie_sentiment[0][1]
        for i in range(1, len(movie_sentiment) - 1):
            if movie_sentiment[i][1] > 0:
                response += "you liked {} and ".format(self.movies[movie_sentiment[i][0]][0])
                self.user_movies[movie_sentiment[i][0]] = movie_sentiment[i][1]
            elif movie_sentiment[i][1] == 0:
                response += "I am not sure whether you liked {} and ".format(self.movies[movie_sentiment[i][0]][0])
                unsure = True
            else:
                response += "you did not like {} and ".format(self.movies[movie_sentiment[i][0]][0])
                self.user_movies[movie_sentiment[i][0]] = movie_sentiment[i][1]

        if movie_sentiment[-1][1] > 0:
            response += "you liked {}.".format(self.movies[movie_sentiment[-1][0]][0])
            self.user_movies[movie_sentiment[-1][0]] = movie_sentiment[-1][1]
        elif movie_sentiment[-1][1] == 0:
            response += "I am not sure whether you liked {}.".format(self.movies[movie_sentiment[-1][0]][0])
            unsure = True
        else:
            response += "you did not like {}.".format(self.movies[movie_sentiment[-1][0]][0])
            self.user_movies[movie_sentiment[-1][0]] = movie_sentiment[-1][1]
        if unsure:
            response += " Please clarify what you felt about the movies I wasn't sure about, thanks!"
        if len(self.user_movies) > 4:
            self.recommending = True
        return response

    def recommend_helper(self, index, sentiment):
        self.user_movies[index] = sentiment
        user_ratings = np.zeros(9125)
        for movie_id in self.user_movies:
            user_ratings[movie_id] = self.user_movies[movie_id]
        recommendations = self.recommend(user_ratings, self.ratings, k=1)
        if sentiment > 0:
            response = "I see, you liked {}. Thank you! I recommend {}. Would you like another recommendation?".format(
                self.movies[index][0], self.movies[recommendations[0]][0])
        else:
            response = "I see, you didn't like {}. Thank you! I recommend {}. Would you like another recommendation?".format(
                self.movies[index][0], self.movies[recommendations[0]][0])
        self.recommending = True
        return response

    def continue_recommend(self, index):
        user_ratings = np.zeros(9125)
        for movie_id in self.user_movies:
            user_ratings[movie_id] = self.user_movies[movie_id]
        recommendations = self.recommend(user_ratings, self.ratings, k=index + 1)
        line_0 = "I recommend {}. Would you like another one?".format(self.movies[recommendations[index]][0])
        line_1 = "I think you would like {}. Would you like to continue?".format(self.movies[recommendations[index]][0])
        line_2 = "I bet you would enjoy {}. More recommendations?".format(self.movies[recommendations[index]][0])
        lines = [line_0, line_1, line_2]
        num = np.random.randint(0, 2)
        response = lines[num]
        return response

    def end(self):
        response = "Thank you! Please type :quit to quit the bot."
        self.rec_count = 1
        self.recommending = False
        self.user_movies = {}
        return response

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        text = self.preprocess(line)

        if self.binary_clarifying[0]:
            response = self.binary_clarification_follow_up(text.lower())
        elif self.multiple_clarifying[0]:
            response = self.multiple_clarification_follow_up(text)
        else:
            if self.recommending:

                if text.lower() not in self.affirm and text.lower() not in self.negate:
                    response = "Sorry, I do not understand. Please answer Yes or No."
                else:
                    if text.lower() in self.affirm:
                        response = self.continue_recommend(self.rec_count)
                        self.rec_count += 1
                    else:
                        response = self.end()

            else:
                titles = self.extract_titles(text)
                if len(titles) == 0:
                    response = self.invalid_input(text)
                elif len(titles) > 1:
                    pair = self.extract_sentiment_for_movies(text)
                    response = self.multiple_query(pair)
                    if self.recommending:
                        response += self.continue_recommend(0)

                else:
                    movie = titles[0]
                    sentiment = self.extract_sentiment(text)
                    movie_id = self.find_movies_by_title(movie)
                    if len(movie_id) == 0:
                        response = self.no_such_movie(movie, sentiment)
                    elif len(movie_id) > 1:
                        response = self.multiple_matches(movie_id, sentiment)
                    else:
                        index = movie_id[0]
                        if len(self.user_movies) < 4:
                            if index in self.user_movies:
                                response = self.correction(index, sentiment)
                            else:
                                response = self.echo_sentiment(index, sentiment)
                        else:
                            response = self.recommend_helper(index, sentiment)
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, and extract_sentiment_for_movies
        methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        punctuations = ['.', ',', '?', '!']
        i = -1
        while text[i] in punctuations:
            text = text[:-1]
            i -= 1
        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        if preprocessed_input.find("\"") >= 0:
            spl = preprocessed_input.split("\"")
            movies = []
            for i in range(1, len(spl), 2):
                movies.append(spl[i])
            return movies
        else:
            prev = ["like", "see", "enjoy", "think", "love", "thought", "liked", "saw", "enjoyed", "loved"]
            nxt = ["was", "is", "be", "much"]
            p = porter_stemmer.PorterStemmer()
            stemmed_prev = []
            stemmed_nxt = []
            for word in prev:
                stemmed_prev.append(p.stem(word))
            for word in nxt:
                stemmed_nxt.append(p.stem(word))
            spl = preprocessed_input.lower().split(" ")
            stemmed_spl = []
            for word in spl:
                stemmed_spl.append(p.stem(word))
            is_prev = False
            is_nxt = False
            for word in stemmed_prev:
                if word in stemmed_spl:
                    is_prev = True
                    id_prev = stemmed_spl.index(word)
                    break
            for word in stemmed_nxt:
                if word in stemmed_spl:
                    is_nxt = True
                    id_nxt = stemmed_spl.index(word)
                    break
            if not is_nxt:
                i = 0
                a = p.stem("a")
                lot = p.stem("lot")
                so = p.stem("so")
                much = p.stem("much")
                while (i < len(stemmed_spl)):
                    if a in stemmed_spl[i:]:
                        a_id = stemmed_spl[i:].index(a) + i
                        if a_id < len(stemmed_spl) - 1:
                            if stemmed_spl[a_id + 1] == lot:
                                is_nxt = True
                                id_nxt = a_id
                                break
                        i = a_id + 1
                    else:
                        break
                if not is_nxt:
                    i = 0
                    while (i < len(stemmed_spl)):
                        if so in stemmed_spl[i:]:
                            so_id = stemmed_spl[i:].index(so) + i
                            if so_id < len(stemmed_spl) - 1:
                                if stemmed_spl[so_id + 1] == much:
                                    is_nxt = True
                                    id_nxt = so_id
                                    break
                            i = so_id + 1
                        else:
                            break
            title = ""
            if not is_prev and not is_nxt:
                return []
            if is_prev and not is_nxt:
                title = " ".join(spl[id_prev + 1:])
            if not is_prev and is_nxt:
                title = " ".join(spl[:id_nxt])
            if is_prev and is_nxt:
                title = " ".join(spl[id_prev + 1:id_nxt])
            if "!" in title:
                return [title[:-1]]
            if "?" in title:
                return [title[:-1]]
            if "." in title:
                return [title[:-1]]
            return [title]

    def extract_alternate_title(self, raw_title, flag):
        curr = raw_title
        temp = curr
        year = curr.split()[-1]
        result = []
        count = raw_title.count('(')
        if count > 1:
            for i in range(count - 1):
                curr = temp
                start = curr.find('(')
                end = curr.find(')')
                curr_list = curr[start:end + 1].split()
                final_list = []
                if len(curr_list) == 1:
                    final_list = [curr_list[0][1:-1]]
                else:
                    first = curr_list[0][1:]
                    last = curr_list[-1][:-1]
                    if len(curr_list) == 2:
                        final_list = [first, last]
                    else:
                        final_list.append(first)
                        for j in range(len(curr_list) - 2):
                            final_list.append(curr_list[j + 1])
                        final_list.append(last)
                if final_list[0] == 'a.k.a.':
                    final = final_list[1:]
                else:
                    final = final_list
                if flag:
                    final.append(year)
                result.append(final)
                temp = curr[start:end + 1]
        return result

    def disambiguation1(self, query, index):
        new_index = []
        for term in index:
            if term[-1] in self.punctuations:
                new_term = term[:-1]
            else:
                new_term = term
            new_index.append(new_term)
        i = 0
        j = 0
        k = 1
        while i < len(query) and j < len(new_index):
            if query[i][-1] in self.punctuations:
                new_word = query[i][:-1]
            else:
                new_word = query[i]
            if new_word == new_index[j]:
                i += 1
                j += 1
                k = j
            else:
                i = 0
                j = k
                k += 1
        if i == len(query):
            return True
        else:
            return False

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        ids = []
        T_spl = []
        t_spl = title.lower().split()
        articles = ['the', 'an', 'a', 'la', 'les', 'le']
        if t_spl[0].lower() in articles:
            if t_spl[-1][0] == '(':
                T_spl = t_spl[1:-1]
            else:
                T_spl = t_spl[1:]
            T_spl[-1] += ","
            T_spl.append(t_spl[0])
            if t_spl[-1][0] == '(':
                T_spl.append(t_spl[-1])
        else:
            T_spl = t_spl.copy()
        movies = util.load_titles("data/movies.txt")
        if T_spl[-1][0] == '(' and len(T_spl[-1]) == 6 and T_spl[-1][1:5].isdecimal():
            for i in range(len(movies)):
                raw_title = movies[i][0].lower()
                if raw_title.count("(") > 1:
                    matchable_title = raw_title[:raw_title.find("(")].split()
                    matchable_title.append(raw_title.split()[-1])
                    if self.disambiguation1(T_spl, matchable_title):
                        ids.append(i)

                    else:
                        alt_title_list = self.extract_alternate_title(raw_title, True)
                        for name in alt_title_list:
                            if self.disambiguation1(T_spl, matchable_title):
                                ids.append(i)

                else:
                    if self.disambiguation1(T_spl, raw_title.split()):
                        ids.append(i)

        else:
            for i in range(len(movies)):
                raw_title = movies[i][0].lower()
                id = raw_title.find("(")
                if self.disambiguation1(T_spl, raw_title[:id].split()):
                    ids.append(i)

                if raw_title.count('(') > 1:
                    alt_title_list = self.extract_alternate_title(raw_title, False)
                    for name in alt_title_list:
                        if self.disambiguation1(T_spl, name):
                            ids.append(i)

        return ids

    negation_words = ["no", "not", "none", "nobody", "nothing", "neither", "nowhere", "never", "hardly",
                      "scarcely", "barely", "seldom", "doesn't", "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't",
                      "won't", "can't",
                      "don't", "didn't", "haven't", "hadn't", "hasn't", "cannot", "mightn't", "needn't"]

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        As an optional creative extension, return -2 if the sentiment of the
        text is super negative and +2 if the sentiment of the text is super
        positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        strong_words = ["horrible", "terrible", "loved", "really", "reeally", "hated", "brilliant"]
        spl = preprocessed_input.split()
        dict = util.load_sentiment_dictionary("data/sentiment.txt")
        p = porter_stemmer.PorterStemmer()
        stemmed_dict = {}
        for key in dict:
            stemmed_dict[p.stem(key)] = dict[key]
        conv = {"pos": 1, "neg": -1}
        score = 0
        flag = False
        for i in range(len(spl)):
            if spl[i][0] == "\"":
                flag = True
            if flag:
                if spl[i][-1] == "\"":
                    flag = False
                continue
            word = spl[i].lower()
            if not word[-1].isalnum():
                new_word = word[:-1]
                word = new_word
            stemmed = p.stem(word)
            sub_score = 0
            if stemmed not in stemmed_dict:
                continue
            sub_score = conv[stemmed_dict[stemmed]]
            if self.creative and word in strong_words:
                sub_score *= 2
            for j in range(1, min(i + 1, 4)):
                if spl[i - j].lower() in self.negation_words:
                    sub_score *= -1
            score += sub_score
        res = 0
        if score > 0:
            res = 1
        if score < 0:
            res = -1
        if self.creative:
            if score > 1:
                res = 2
            if score < -1:
                res = -2
        return res

    def extract_sentiment_for_movies(self, preprocessed_input):
        """Creative Feature: Extracts the sentiments from a line of
        pre-processed text that may contain multiple movies. Note that the
        sentiments toward the movies may be different.

        You should use the same sentiment values as extract_sentiment, described

        above.
        Hint: feel free to call previously defined functions to implement this.

        Example:
          sentiments = chatbot.extract_sentiment_for_text(
                           chatbot.preprocess(
                           'I liked both "Titanic (1997)" and "Ex Machina".'))
          print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a list of tuples, where the first item in the tuple is a movie
        title, and the second is the sentiment in the text toward that movie
        """
        sentiments = []
        pos_connectors = ['and', 'or', 'both']
        neg_connectors = ['but']
        pos = False
        neg = False
        neg_pos = []
        input_list = preprocessed_input.split()
        for word in input_list:
            if word in pos_connectors:
                pos = True
            if word in neg_connectors:
                neg = True
                neg_pos.append(preprocessed_input.find(word))
        titles = self.extract_titles(preprocessed_input)
        first_position = preprocessed_input.find(titles[0])
        first_len = len(titles[0])
        first_substring = preprocessed_input[:first_position + first_len]
        sentiment = self.extract_sentiment(first_substring)
        if sentiment == 0:
            second_substring = preprocessed_input[:preprocessed_input.find(titles[1])]
            sentiment = self.extract_sentiment(second_substring)
        if (pos and not neg) or (not pos and not neg):
            for title in titles:
                sentiments.append((title, sentiment))
        elif neg:
            sentiments.append((titles[0], sentiment))
            j = 1
            for i in range(len(neg_pos)):
                while preprocessed_input.find(titles[j]) < neg_pos[i]:
                    sentiments.append((titles[j], sentiment))
                    j += 1
                sentiment *= -1
            for k in range(j, len(titles)):
                sentiments.append((titles[k], sentiment))
        else:
            for k in range(len(titles)):
                position = preprocessed_input.find(titles[k])
                length = len(titles[k])
                if k == 0:
                    substring = preprocessed_input[:position + length]
                else:
                    prev = preprocessed_input.find(titles[k - 1])
                    prev_length = len(titles[k - 1])
                    substring = preprocessed_input[prev + prev_length:position + length]
                sentiment = self.extract_sentiment(substring)
                sentiments.append((titles[k], sentiment))
        return sentiments

    def extract_valid_title(self, raw_title):
        par_loc = raw_title.find("(")
        if par_loc >= 6 and raw_title[par_loc - 6:par_loc] == ", the ":
            return "the " + raw_title[:par_loc - 6]
        elif par_loc >= 5 and raw_title[par_loc - 5:par_loc] == ", an ":
            return "an " + raw_title[:par_loc - 5]
        elif par_loc >= 4 and raw_title[par_loc - 4:par_loc] == ", a ":
            return "a " + raw_title[:par_loc - 4]
        else:
            return raw_title[:par_loc - 1]

    def edit_distance(self, s1, s2):
        dist = np.zeros((len(s1), len(s2)))
        for i in range(len(s1)):
            dist[i, 0] = i
        for j in range(len(s2)):
            dist[0, j] = j
        for i in range(1, len(s1)):
            for j in range(1, len(s2)):
                cost = 0
                if s1[i] != s2[j]:
                    cost = 2
                dist[i, j] = min([dist[i - 1, j - 1] + cost, dist[i - 1, j] + 1, dist[i, j - 1] + 1])
        return dist[len(s1) - 1, len(s2) - 1]

    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least
        edit distance from the provided title, and with edit distance at most
        max_distance.

        - If no movies have titles within max_distance of the provided title,
        return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given
        title than all other movies, return a 1-element list containing its
        index.
        - If there is a tie for closest movie, return a list with the indices
        of all movies tying for minimum edit distance to the given movie.

        Example:
          # should return [1656]
          chatbot.find_movies_closest_to_title("Sleeping Beaty")

        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title
        and within edit distance max_distance
        """
        ids = []
        movies = util.load_titles("data/movies.txt")
        for i in range(len(movies)):
            # print(i)
            raw_title = movies[i][0].lower()
            # print(raw_title)
            if raw_title.count("(") > 1:
                val_titles = []
                val_titles.append(self.extract_valid_title(raw_title))
                sub_title = raw_title[raw_title.find("(") + 1:].replace(")", "")
                val_titles.append(self.extract_valid_title(sub_title))
                # print(val_titles)
                min_dist = max_distance + 1
                for val_title in val_titles:
                    min_dist = min(min_dist, self.edit_distance("#" + title.lower(), "#" + val_title))
            else:
                val_title = self.extract_valid_title(raw_title)
                # print(val_title)
                min_dist = self.edit_distance("#" + title.lower(), "#" + val_title)

            if min_dist <= max_distance:
                ids.append((min_dist, i))
        if len(ids) == 0:
            return []
        arr = np.array(sorted(ids)[:min(3, len(ids))])[:, 1]
        return arr.astype(int)

    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (eg. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)

        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it
        should return a list with the indices it could be referring to (to
        continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the
        given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by
        the clarification
        """
        movies = util.load_titles("data/movies.txt")
        c = {}
        for id in candidates:
            c[id] = movies[id][0]
        positions = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eigth', 'ninth', 'tenth']
        new = ['recent', 'latest', 'newest']
        old = ['oldest', 'earliest']
        no_cap = ['a', 'an', 'the', 'for', 'and', 'nor', 'but', 'or', 'yet', 'so', 'at', 'around', 'by', 'after',
                  'along', 'from', 'of', 'on', 'to', 'with', 'without']
        if clarification.isnumeric():
            if len(clarification) == 4:
                for key, val in c.items():
                    if clarification in val:
                        return [key]
            else:
                for key, val in c.items():
                    lst = val.split()
                    if clarification in lst:
                        return [key]
                return [candidates[int(clarification) - 1]]
        else:
            for key, val in c.items():
                if clarification in val:
                    return [key]
            for i in range(len(positions)):
                if positions[i] in clarification:
                    return [candidates[i]]
            words = clarification.split()
            curr = ''
            for word in words:
                if word[0].isupper():
                    if curr != '':
                        curr += ' '
                    curr += word
                    continue
                if curr != '' and (word in no_cap):
                    curr += (' ' + word)
                    continue
                if curr != '' and word[0].islower():
                    break
            if curr != '':
                for key, val in c.items():
                    if curr in val:
                        return [key]
            for word in new:
                for i in range(len(words)):
                    if word == words[i]:
                        if words[i - 1] == 'least':
                            return [candidates[len(candidates) - 1]]
                        return [candidates[0]]
            for word in old:
                for i in range(len(words)):
                    if word == words[i]:
                        if words[i - 1] == 'least':
                            return [candidates[0]]
                        return [candidates[len(candidates) - 1]]
        return []

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        binarized_ratings = np.zeros_like(ratings)
        binarized_ratings[ratings > threshold] = 1
        binarized_ratings[(0 < ratings) & (ratings <= threshold)] = -1
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ids = np.where((u != 0) & (v != 0))[0]

        if len(ids) == 0:
            return 0
        similarity = u[ids] @ v[ids] / (np.linalg.norm(u) * np.linalg.norm(v))

        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        # Populate this list with k movie indices to recommend to the user.
        rated = np.where(user_ratings != 0)[0]
        unrated = np.where(user_ratings == 0)[0]
        preds = []
        for target in unrated:
            den = 0.0
            num = 0.0
            for item in rated:
                sim = self.similarity(ratings_matrix[item], ratings_matrix[target])
                den += np.abs(sim)
                num += sim * user_ratings[item]
            if den == 0:
                preds.append((0, den, target))
            else:
                preds.append((num / den, den, target))
        recommendations = [int(x) for x in np.array(sorted(preds, reverse=True)[:k])[:, 2]]
        return recommendations

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """
        return """
        Hello! I am movie bot and I would like to recommend you some movies. 
        I would need 5 movies for reference before making my recommendation.
        Let's get started! 
        """


if __name__ == '__main__':
    chat = Chatbot()
    str = "I enjoy"
    """
    titles = chat.extract_titles(str)
    for title in titles:
        print("raw_title: ", title)
        print(chat.find_movies_by_title(title))
    """
    print(chat.extract_sentiment(str))
    """
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
    """
