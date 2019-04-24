import numpy as np
from scipy.stats import t

def activity_filter(ans_dates: np.ndarray, cur_date: np.datetime64, tail_prob: np.double=0.05,
                    min_days: np.double=0.5, max_days: np.double=15):
    """
    Check whether a professional is currently active or not
    by estimating how likely distance from current date to
    previous answer date is to come from t-distribution with
    parameters estimated from professional's answer dates.
    
    :param ans_dates: ndarray of professional's answer dates
    :param cur_date: np.datetime64 object containing current date
    :param tail_prob: tail probability of t-distribution
    :param min_days: minimum number of days after previous answer date
    :param max_days: maximum number of days after previous answer date
    """
    # get realization of some random variable,
    # which is distance between current date and previos answer date
    realiz = (cur_date - ans_dates[-1]) / np.timedelta64(1, 'D')
    
    # compute intervals between answers if there is at least two answers so far
    # (the first date in ans_dates is registration date)
    if ans_dates.size >= 3:
        int_lens = np.diff(ans_dates) / np.timedelta64(1, 'D')
    else:
        # check if registration or previous answer was
        # more than min_days and less than max_days ago
        return True if (realiz > min_days and realiz < max_days) else False
    
    # clip interval lengths to be in range [min_days, max_days]
    int_lens = np.clip(int_lens, min_days, max_days)
    
    # estimate parameters of t-distribution
    df = int_lens.size - 1
    loc = int_lens.mean()
    scale = max(int_lens.std(ddof=1), min_days)
    
    # estimate probability of this random variable coming from t-distibution
    prob = t.cdf(realiz, df=df, loc=loc, scale=scale)    
    if prob > 0.5:
        prob = 1 - prob
    
    # if this probability is greater than probability of coming from one
    # tail of t-distribution, or previous answer was less than max_days ago,
    # and also it's been at least min_days since previous answer,
    # than conclude that professional is active
    if (prob > tail_prob / 2 or realiz < max_days) and realiz > min_days:
        return True
    return False


def spam_filter(question_id: str, email_ques: np.ndarray, previous_email_date: np.datetime64,
                   cur_date: np.datetime64, min_days: np.double=0.5):
    """
    Do not send an email if previous email was sent less than min_days ago
    or if professional was already email-notified about the question.
    
    :param que: question about which email notification is going to be sent
    :param email_ques: ndarray of email-notified questions
    :param previous_email_date: np.datetime64 object containing previous email date
    :param cur_date: np.datetime64 object containing current date
    :param min_days: minimum number of days after previous email notification
    """
    if (cur_date - previous_email_date) / np.timedelta64(1, 'D') < min_days or question_id in email_ques:
        return False
    return True


def email_filter(que: str, email_ques: np.ndarray, email_dates: np.ndarray,
                 ans_ques: np.ndarray, ans_dates: np.ndarray,
                 cur_date: np.datetime64, offset_days: np.double,
                 min_days: np.double=0.5, max_days: np.double=7, thresh: np.double=0.1):
    """
    Makes a decision about whether to send email to active professional or not
    by computing the coefficient based on fraction of email-notified questions
    that were answered and on distribution of intervals between
    email notification and answer date.
    
    :param que: question about which email notification is going to be sent
    :param email_ques: ndarray of email-notified questions
    :param email_dates: ndarray of dates email notifications where sent
    :param ans_ques: ndarray of answered questions
    :param ans_dates: ndarray of answer dates
    :param cur_date: np.datetime64 object containing current date
    :param offset_days: period (in days) during which we want que to be answered
    :param min_days: minimum number of days after previous email notification
    :param max_days: number of days on which to clip interval lengths
    :param thresh: threshold for F1 score
    """
    # do not send email if previous email was sent less than min_days ago
    # or if professional was already email-notified about this question
    if not spam_filter(que, email_ques, email_dates.max(), cur_date, min_days):
        return False
    
    # select answered email-notified questions, and also email and answer dates assosiated to them
    ques, email_idx, ans_idx = np.intersect1d(email_ques, ans_ques, assume_unique=False, return_indices=True)
    email_dates = email_dates[email_idx]
    ans_dates = ans_dates[ans_idx]
    
    # do not send email if less than 2 email-notified questions were answered
    if ques.size < 2:
        return False
    
    # compute fraction of email-notified questions that were answered
    email_frac = ques.size / email_ques.size
    
    # compute intervals between email notification and answer date
    # and clip them to be in range [min_days, max_days]
    int_lens = (ans_dates - email_dates) / np.timedelta64(1, 'D') 
    int_lens = np.clip(int_lens, min_days, max_days)
    
    # estimate parameters of t-distribution
    df = int_lens.size - 1
    loc = int_lens.mean()
    scale = max(int_lens.std(ddof=1), min_days)
    
    # estimate probability of answering an email-notified question within offset_days period
    ans_prob = t.cdf(offset_days, df=df, loc=loc, scale=scale)
    
    # compute F1 score based on email_frac and anwer_prob
    score = 2 * email_frac * ans_prob / (email_frac + ans_prob)
    
    # send email if F1 score is larger than threshold
    return score > thresh