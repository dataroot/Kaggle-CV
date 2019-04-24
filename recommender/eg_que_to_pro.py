import numpy as np
import pandas as pd

from activity import activity_filter, spam_filter


def send_quesionts_to_professional(pro_sample_dict, pro_answer_dates, questions,
                                   answers, pro_email_ques, current_date=np.datetime64('now'), 
                                   top_content=20, min_days=7):
    eps_1 = 0.01
    eps_2 = 0.5
    eps_3 = 0.3
    
    # Get top n suggested questions
    pro_sample_df, pro_sample_tags = Formatter.convert_pro_dict(pro_sample_dict)
    tmp = predictor.find_ques_by_pro(pro_sample_df, questions, answers, pro_sample_tags, top=top_content)
    content_result = formatter.get_que(tmp)
    que_ids = content_result['questions_id'].values
    
    # Get answer dates for professional
    pro_id = pro_sample_df['professionals_id'].iloc[0]
    answer_dates = pro_answer_dates.loc[pro_id].values
    
    # Check if professional is active
    is_active = activity_filter(answer_dates, current_date)
    
    # Exclude dates that are greater than current date
    try:
        email_ques = pro_email_ques['questions_id'].loc[pro_id].values
        email_dates = pro_email_ques['emails_date_sent'].loc[pro_id].values

        email_ques, email_dates = email_ques[email_dates < current_date], email_dates[email_dates < current_date]
    except:
        email_ques = []
        email_dates = []

    if not email_ques:
        mask = [True] * len(que_ids)
    else:
        mask = []
        for que_id in que_ids:
            mask.append(spam_filter(que_id, email_ques, email_dates.max(), cur_date, min_days=min_days))
    
    # Divide mails to spam / not spam
    mask = np.array(mask)
    passed_questions = que_ids[mask]
    spam_questions = que_ids[~mask]
    
    explore_questions = []
    
    # epsilon greedy
    if is_active:
        if passed_questions.size > 0:
            for sq in spam_questions:
                if np.random.rand() < eps_1:
                    explore_questions.append(sq)
    else:
        if passed_questions.size > 0:
            e_cond = eps_2
        else:
            e_cond = eps_3
    
        for sq in spam_questions:
            if np.random.rand() < e_cond:
                explore_questions.append(sq)
            
    final_q_ids = np.append(passed_questions, explore_questions)       
    final_df = content_result[content_result['questions_id'].isin(final_q_ids)]
    
    return final_df
