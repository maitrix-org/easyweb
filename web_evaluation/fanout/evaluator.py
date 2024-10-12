import json

import rouge_score
import rouge_score.scoring
from helpers import answer_in_text, str_answer
from models import (
    RougeScore,
    RougeScorePart,
)
from rouge_score.rouge_scorer import RougeScorer

ROUGE_TYPES = ('rouge1', 'rouge2', 'rougeL')


class FanOutQAEvaluator:
    def __init__(self, test_data_path):
        self.test_data = json.load(open(test_data_path))
        self.questions = self.test_data[:20]
        # self.question2id = {q['question']: q['id'] for q in self.questions}
        self.questions_by_id = {q['id']: q for q in self.questions}

        self.rouge = RougeScorer(ROUGE_TYPES, use_stemmer=True)

    def evaluate_batch(self, evaluator_log_paths):
        records = []
        answers = []
        for evaluator_log_path in evaluator_log_paths:
            idx, question, answer, outcome = self._load_evaluator_log(
                evaluator_log_path
            )
            question_id = self.questions[idx]['id']
            question = self.questions[idx]['question']
            if outcome == 'Response Returned':
                answers.append({'id': question_id, 'answer': answer})
            records.append(
                {
                    'id': question_id,
                    'question': question,
                    'answer': answer,
                    'outcome': outcome,
                }
            )

        self.answers = answers
        self.answers_by_id = {r['id']: r for r in answers}

        result, raw_scores = self._score_accuracy()
        scores, raw_scores = self._score_rouge()

        row = {
            'acc_loose': result['loose'],
            'acc_strict': result['strict'],
            'rouge1': scores.rouge1.fscore,
            'rouge2': scores.rouge2.fscore,
            'rougeL': scores.rougeL.fscore,
        }
        return row, records

    def _load_evaluator_log(self, file_path, verbose=False):
        data = json.load(open(file_path))

        idx = int(file_path.split('_')[-2])
        goal = ''
        response = None
        started = False

        for message in data:
            if message.get('action') == 'message':
                response = message['message']
            elif message.get('action') == 'browse_interactive':
                if not started:
                    started = True
                    continue

                step_record = json.loads(message['args']['thought'])
                if step_record.get('obs'):
                    goal = step_record['obs']['goal']

        outcome = 'Response Returned'
        if response in [
            'Too many errors encountered. Task failed.',
            'It seems I am stuck. Could you help me out?',
        ]:
            outcome = 'Action Error'
        elif response == 'Error encountered when browsing.':
            outcome = 'Browsing Error'
        elif response is None:
            outcome = 'Max Time or Steps Reached'

        goal = goal.replace('Using Google, ', '')

        return idx, goal, response, outcome

    def _get_qa_pairs(self, only_score_answered=False):
        """Yield pairs of questions and answers to score.
        The answer may be None if there is no answer for a given question and ``only_score_answered`` is False.
        """
        if only_score_answered:
            for a in self.answers:
                q = self.questions_by_id.get(a['id'])
                yield q, a
        else:
            for q in self.questions:
                a = self.answers_by_id.get(q['id'])
                yield q, a

    def _score_accuracy(self, **kwargs):
        """Get the loose and strict accuracy scores for the loaded qs and as."""
        eval_len = len(self.questions)

        raw_scores = {}  # qid -> score
        accs = []
        n_perfect = 0
        for q, a in self._get_qa_pairs(**kwargs):
            if a is None:
                accs.append(0)
                raw_scores[q['id']] = 0
                continue
            result = answer_in_text(q['answer'], a['answer'])
            accs.append(result['score'])
            raw_scores[q['id']] = result['score']
            if result['found']:
                n_perfect += 1

        assert len(accs) == eval_len
        assert len(raw_scores) == eval_len
        avg_acc = sum(accs) / eval_len
        pct_perfect = n_perfect / eval_len
        return dict(loose=avg_acc, strict=pct_perfect), raw_scores

    def _score_rouge(self, **kwargs):
        """Get the ROUGE-1, ROUGE-2, and ROUGE-L scores (P/R/F1) for the loaded qs and as."""
        eval_len = len(self.questions)

        raw_scores = {}  # qid -> RougeScore
        scores = {t: [] for t in ROUGE_TYPES}  # rouge_type -> list[Score]
        for q, a in self._get_qa_pairs():
            if a is None:
                for score in scores.values(**kwargs):
                    score.append(rouge_score.scoring.Score(0, 0, 0))
                raw_scores[q['id']] = RougeScore(
                    **{
                        k: RougeScorePart(precision=0, recall=0, fscore=0)
                        for k in ROUGE_TYPES
                    }
                )
                continue
            results = self.rouge.score(str_answer(q['answer']), str_answer(a['answer']))
            for k, v in results.items():
                scores[k].append(v)
            raw_scores[q['id']] = RougeScore(
                **{
                    k: RougeScorePart(
                        precision=v.precision, recall=v.recall, fscore=v.fmeasure
                    )
                    for k, v in results.items()
                }
            )

        assert all(len(v) == eval_len for v in scores.values())
        assert len(raw_scores) == eval_len
        out = {}
        for k, v in scores.items():
            avg_precision = sum(s.precision for s in v) / eval_len
            avg_recall = sum(s.recall for s in v) / eval_len
            avg_fscore = sum(s.fmeasure for s in v) / eval_len
            out[k] = RougeScorePart(
                precision=avg_precision, recall=avg_recall, fscore=avg_fscore
            )
        return RougeScore(**out), raw_scores
