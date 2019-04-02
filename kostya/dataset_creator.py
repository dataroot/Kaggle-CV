import numpy as np
import pandas as pd

import re, os


class DatasetCreator:
    """
    Class that imports initial datasets and creates additional datasets for convenience
    """
    
    def __init__(self, data_path='../../data/', created=False):
        # Add data_path to class properties
        self.data_path = data_path
        
        # Import all initial datasets
        self.emails = pd.read_csv(data_path + 'emails.csv')
        self.questions = pd.read_csv(data_path + 'questions.csv')
        self.professionals = pd.read_csv(data_path + 'professionals.csv')
        self.comments = pd.read_csv(data_path + 'comments.csv')
        self.tag_users = pd.read_csv(data_path + 'tag_users.csv')
        self.group_memberships = pd.read_csv(data_path + 'group_memberships.csv')
        self.tags = pd.read_csv(data_path + 'tags.csv')
        self.students = pd.read_csv(data_path + 'students.csv')
        self.groups = pd.read_csv(data_path + 'groups.csv')
        self.tag_questions = pd.read_csv(data_path + 'tag_questions.csv')
        self.matches = pd.read_csv(data_path + 'matches.csv')
        self.answers = pd.read_csv(data_path + 'answers.csv')
        self.school_memberships = pd.read_csv(data_path + 'school_memberships.csv')
        
        if created:
            # Load additional datasets from disk
            self.qa_data = pd.read_csv(self.data_path + 'qa_data.csv')
            self.prof_data = pd.read_csv(self.data_path + 'prof_data.csv')
            self.stud_data = pd.read_csv(self.data_path + 'stud_data.csv')
        else:
            # Create additional datasets and save them to disk
            self.additional_datasets_creation()
    
    
    def additional_datasets_creation(self):
        """
        Creates additional datasets for futher processing and save them to disk.
        """
        # Create temporary dataset for further processing
        all_data = self.all_data_creation()
        
        # Create question-answer pairs dataset called qa_data
        self.qa_data = self.qa_data_creation(all_data)
        
        # Create dataset called prof_data compirising data of professionals
        # who answered at least one question
        self.prof_data = self.prof_data_creation(all_data)
        
        # Create dataset called stud_data compirising data of students
        # who asked at least one answered question
        self.stud_data = self.stud_data_creation(all_data)
        
        # Save new datasets to disc
        self.qa_data.to_csv(self.data_path + 'qa_data.csv', index=False)
        self.prof_data.to_csv(self.data_path + 'prof_data.csv', index=False)
        self.stud_data.to_csv(self.data_path + 'stud_data.csv', index=False)
    
    
    def all_data_creation(self):
        """
        Merges questions, answers, professionals and students datasets
        to get temporary dataset for further processing
        """
        # Merge questions with answers and delete not answered questions
        all_data = self.questions.merge(self.answers, left_on='questions_id', right_on='answers_question_id')
        
        # Merge with professionals and students (students asked, professionals answered)
        # Maybe change this in the future by taking care of professional who change status to students and vise versa
        all_data = all_data.merge(self.professionals, left_on='answers_author_id', right_on='professionals_id')
        all_data = all_data.merge(self.students, left_on='questions_author_id', right_on='students_id')
        
        # Transform dates from string representation to datetime object
        all_data.answers_date_added = pd.to_datetime(all_data.answers_date_added)
        all_data.questions_date_added = pd.to_datetime(all_data.questions_date_added)
        
        # Add questions_age feature, which represents amount of time
        # from question emergence to a particular answer to that question
        all_data['questions_age'] = all_data.answers_date_added - all_data.questions_date_added
        
        # Delete html tags and extra spaces from question and answer body
        all_data.questions_body = (all_data.questions_body
                                   .apply(lambda x: re.sub(r'(<[^>]*[/]?>|[\r]?\n)', ' ', str(x)))
                                   .apply(lambda x: re.sub(r' +', ' ', x).strip()))
        all_data.answers_body = (all_data.answers_body
                                 .apply(lambda x: re.sub(r'(<[^>]*[/]?>|[\r]?\n)', ' ', str(x)))
                                 .apply(lambda x: re.sub(r' +', ' ', x).strip()))
        
        # Count the number of words in question and answer body and add two new features
        all_data['questions_body_length'] = all_data.questions_body.apply(lambda x: len(x.split(' ')))
        all_data['answers_body_length'] = all_data.answers_body.apply(lambda x: len(x.split(' ')))
        
        return all_data
    
    
    def qa_data_creation(self, all_data):
        """
        Creates question-answer pairs dataset called qa_data_data
        """
        # Temporary qa_data representation
        qa_data = all_data.copy()
        
        # Select only unique professionals
        temp = qa_data[['professionals_id', 'answers_date_added', 'answers_id']]
        prof_unique = pd.DataFrame(temp.professionals_id.unique(), columns=['professionals_id'])
        prof_unique = prof_unique.merge(self.professionals, how='left', on='professionals_id')
        
        # For every professional add a "dummy" question with answer date being professional's registration date
        prof_unique['answers_id'] = list(None for _ in range(prof_unique.shape[0]))
        prof_unique['answers_date_added'] = prof_unique['professionals_date_joined']
        prof_unique = prof_unique[['professionals_id', 'answers_date_added', 'answers_id']]
        
        # Add "dummy" questions to all questions
        temp = pd.concat([temp, prof_unique])
        
        # Sort by professionals and answer dates
        temp = temp.sort_values(by=['professionals_id', 'answers_date_added']).reset_index(drop=True)
        
        # Get the sorted representation of the answers_date_added and shift the index down by one
        # so that current question is aligned with previous question answer date
        prev_answer_date = pd.DataFrame({'professionals_prev_answer_date': temp.answers_date_added})
        prev_answer_date.index += 1
        
        # Add the professionals_prev_answer_date column to temp
        temp = temp.merge(prev_answer_date, left_index=True, right_index=True)
        temp.dropna(subset=['answers_id'], inplace=True)
        temp.drop(columns=['professionals_id', 'answers_date_added'], inplace=True)
        
        # Add professionals_prev_answer_date column to qa_data 
        qa_data = qa_data.merge(temp, on='answers_id')
        
        # Transform dates from string representation to datetime object
        qa_data.professionals_prev_answer_date = pd.to_datetime(qa_data.professionals_prev_answer_date)
        
        print(qa_data[qa_data.professionals_id == '003cc21be89d4e42bc4424131a378e86']
              [['answers_date_added', 'professionals_prev_answer_date']].sort_values(by='answers_date_added'))
        
        # Final qa_data representation
        qa_data = qa_data[[
            'students_id', 'questions_id', 'questions_title', 'questions_body',
            'questions_body_length', 'questions_date_added', 'professionals_id',
            'answers_id', 'answers_body', 'answers_date_added', 'professionals_prev_answer_date'
        ]]
        
        return qa_data
    
    
    def prof_data_creation(self, all_data):
        """
        Creates dataset called prof_data compirising data of professionals who answered at least one question
        """
        # Select only professionals who answered at least one question
        active_professionals = pd.DataFrame({'professionals_id': all_data.professionals_id.unique()})
        prof_data = self.professionals.merge(active_professionals, how='right', on='professionals_id')
        
        # Extract state or country from location
        prof_data['professionals_state'] = prof_data['professionals_location'].apply(lambda loc: str(loc).split(', ')[-1])
        
        # Transform dates from string representation to datetime object
        prof_data.professionals_date_joined = pd.to_datetime(prof_data.professionals_date_joined)
        
        # Count the number of answered questions by each professional
        number_answered = all_data[['questions_id', 'professionals_id']].groupby('professionals_id').count()
        number_answered = number_answered.rename({'questions_id': 'professionals_questions_answered'}, axis=1)
        
        # Add professionals_questions_answered feature to prof_data
        prof_data = prof_data.merge(number_answered, left_on='professionals_id', right_index=True)
        
        # Get average question age for every professional among questions he answered
        average_question_age = (
            all_data.groupby('professionals_id')
            .questions_age.mean(numeric_only=False)
        )
        average_question_age = pd.DataFrame({'professionals_average_question_age': average_question_age})
        
        # Add professionals_average_question_age feature to prof_data
        prof_data = prof_data.merge(average_question_age, on='professionals_id')
        
        # Get all emails that every acting professional received
        prof_emails_received = pd.merge(
            prof_data[['professionals_id']], self.emails,
            left_on='professionals_id', right_on='emails_recipient_id')
        
        # Get all questions every acting professional received in emails
        prof_email_questions = prof_emails_received.merge(
            self.matches, how='inner', left_on='emails_id', right_on='matches_email_id')
        
        # Get answered questions about which professionals were notified by email
        questions_answered_from_emails = prof_email_questions.merge(
            self.qa_data[['professionals_id', 'questions_id']],
            left_on=['professionals_id', 'matches_question_id'],
            right_on=['professionals_id', 'questions_id'])
        
        # Count the number of answered questions about which professionals were notified by email
        email_activated = (questions_answered_from_emails
                           .groupby('professionals_id')[['questions_id']].count()
                           .rename(columns={'questions_id': 'professionals_email_activated'}))
        
        # Add professionals_email_activated feature to prof_data
        # This feature is percent of answered questions about which professionals were notified by email
        prof_data = prof_data.merge(email_activated, on='professionals_id', how='left')
        prof_data.professionals_email_activated.fillna(0, inplace=True)
        prof_data.professionals_email_activated /= prof_data.professionals_questions_answered
        
        # Compute average question and answer body length for each professional
        average_question_body_length = all_data.groupby('professionals_id')[['questions_body_length']].mean().reset_index()
        average_answer_body_length = all_data.groupby('professionals_id')[['answers_body_length']].mean().reset_index()
        
        # Add average question and answer body length features to prof_data
        prof_data = (prof_data.merge(average_question_body_length, on='professionals_id')
                     .rename(columns={'questions_body_length': 'professionals_average_question_body_length'}))
        prof_data = (prof_data.merge(average_answer_body_length, on='professionals_id')
                     .rename(columns={'answers_body_length': 'professionals_average_answer_body_length'}))
        
        return prof_data
    
    
    def stud_data_creation(self, all_data):
        """
        Creates dataset called stud_data compirising data of students who asked at least one answered question
        """
        # Select only students who asked at least one answered question
        active_students = pd.DataFrame({'students_id': all_data.students_id.unique()})
        stud_data = self.students.merge(active_students, how='right', on='students_id')
        
        # Extract state or country from location
        stud_data['students_state'] = stud_data['students_location'].apply(lambda loc: str(loc).split(', ')[-1])
        
        # Transform dates from string representation to datetime object
        stud_data.students_date_joined = pd.to_datetime(stud_data.students_date_joined)
        
        # Count the number of asked questions by each student
        number_asked = all_data[['questions_id', 'students_id']].groupby('students_id').count()
        number_asked = number_asked.rename({'questions_id': 'students_questions_asked'}, axis=1)
        
        # Add students_questions_answered feature to stud_data
        stud_data = stud_data.merge(number_asked, left_on='students_id', right_index=True)
        
        # Get average question age for every student among questions he asked that were answered
        average_question_age = (
            all_data.groupby('students_id')
            .questions_age.mean(numeric_only=False)
        )
        average_question_age = pd.DataFrame({'students_average_question_age': average_question_age})
        
        # Add professionals_average_question_age feature to prof_data
        stud_data = stud_data.merge(average_question_age, on='students_id')
        
        # Compute average question and answer body length for each student
        average_question_body_length = all_data.groupby('students_id')[['questions_body_length']].mean().reset_index()
        average_answer_body_length = all_data.groupby('students_id')[['answers_body_length']].mean().reset_index()
        
        # Add average question and answer body length features to stud_data
        stud_data = (stud_data.merge(average_question_body_length, on='students_id')
                     .rename(columns={'questions_body_length': 'students_average_question_body_length'}))
        stud_data = (stud_data.merge(average_answer_body_length, on='students_id')
                     .rename(columns={'answers_body_length': 'students_average_answer_body_length'}))
        
        return stud_data

