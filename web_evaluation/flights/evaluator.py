import json
import re
from datetime import datetime

import numpy as np
from helpers import field_patterns, get_obs, parse_pattern_matches, parser, tqdm_joblib
from joblib import Parallel, delayed
from openai import OpenAI
from prompts import (
    parse_flight_description_prompt,
    parse_flight_message_prompt,
    verify_flight_constraint_prompt,
)
from tqdm import tqdm


class FlightSearchEvaluator:
    def __init__(self, questions_path):
        data = []
        with open(questions_path) as f:
            for line in f:
                data.append(json.loads(line))
        self.questions_data = data
        self.client = OpenAI(
            base_url='http://localhost:8000/v1',
        )

    def evaluate(self, evaluator_log_path, verbose=False):
        example = self._parse_evaluator_log(evaluator_log_path)
        agent_response = self._get_agent_response(example)

        pattern_matches = {
            field: self._parse_example_pattern_matches(example, field)
            for field in field_patterns
        }
        search_criteria = {}
        for field in [
            'ticket_type',
            'seating_class',
            'origin',
            'destination',
            'departure_date',
            'return_date',
            'sorting_criterion',
            'num_passengers',
        ]:
            matches = pattern_matches[field]
            value = matches[-1].replace('\\u200b', '') if matches[-1] else None
            search_criteria[field] = value
        flight_descriptions = list(set(pattern_matches['flight_description']))
        with tqdm_joblib(
            tqdm(
                desc='Parsing flight descriptions',
                total=len(flight_descriptions),
                disable=(not verbose),
            )
        ) as _:
            flight_description_jsons = Parallel(n_jobs=8, backend='threading')(
                delayed(self._parse_flight_description)(d) for d in flight_descriptions
            )
        # flight_description_jsons = [self._parse_flight_description(d) for d in tqdm(flight_descriptions, desc='Parsing flight descriptions')]

        # To evaluate, we first find the closes match between the agent response and the flight descriptions
        # The match is evaluated in terms of the precision of agent response terms vs the flight description terms
        # If any match is found, we consider the response to be grounded and use that to evaluate against the search constraints
        # If no response is found or no match has high enough similarity, we consider the response a failure

        constraints = []
        for row in self.questions_data:
            if row['question'] == example['goal']:
                constraints = row['constraints']
                break
        search_datetime = example['datetimes'][-1]

        grounded = True
        relevant = True
        agent_response_data = []
        if agent_response:
            if verbose:
                print('Parsing flight message')
            flight_message_jsons = self._parse_flight_message(agent_response)

            retrieved_ids = set()
            for flight_message in flight_message_jsons:
                # Measure how well the flight mention matches with the flight descriptions
                flight_precisions = [
                    self._get_flight_precision(flight_message, f)
                    for f in flight_description_jsons
                ]
                # Retrieve the one with highest precision
                max_idx = np.argmax(flight_precisions)
                max_precision = flight_precisions[max_idx]
                max_flight_description = flight_descriptions[max_idx]

                retrieved_before = max_idx in retrieved_ids
                retrieved_ids.add(max_idx)
                precision_threshold = 0.8
                if max_precision < precision_threshold or retrieved_before:
                    grounded = False

                # constraint_responses = []
                # for criterion in tqdm(constraints, desc='Verifying constraints'):
                #     # response one of 3 values: yes, no, unsure
                #     response = self._verify_flight_constraint(search_datetime, search_criteria,
                #                                               flight_description_jsons,
                #                                               max_flight_description, criterion)
                #     constraint_responses.append(response)

                with tqdm_joblib(
                    tqdm(
                        desc='Verifying constraints',
                        total=len(constraints),
                        disable=(not verbose),
                    )
                ) as _:

                    def verify(c, m=max_flight_description):
                        return self._verify_flight_constraint(
                            search_datetime,
                            search_criteria,
                            flight_description_jsons,
                            m,
                            c,
                        )

                    constraint_responses = Parallel(n_jobs=8, backend='threading')(
                        delayed(verify)(c) for c in constraints
                    )

                unsure_constraints = ['eco-friendly', 'CO2 emissions']
                num_yes = sum(
                    [
                        response == 'yes' and c not in unsure_constraints
                        for c, response in zip(constraints, constraint_responses)
                    ]
                )
                num_not_unsure = sum(
                    [
                        response != 'unsure' and c not in unsure_constraints
                        for c, response in zip(constraints, constraint_responses)
                    ]
                )
                constraint_recall = num_yes / num_not_unsure
                recall_threshold = 1.0
                if constraint_recall < recall_threshold:
                    relevant = False

                agent_response = {
                    'mentioned_flight': flight_message,
                    'retrieved_flight_idx': max_idx,
                    'retrieved_flight_precision': max_precision,
                    'constraint_responses': constraint_responses,
                    'constraint_recall': constraint_recall,
                }
                agent_response_data.append(agent_response)
        else:
            grounded = False
            relevant = False

        # Several things to look for:
        # 1. Correct: Grounded and relevant
        # 2. Grounded: Agent response is grounded in the flight descriptions
        # 3. Relevant: Agent response satisfies the search constraints
        # 4. Response Data: For each flight mentioned in the agent response,
        # we store the flight retrieved by it, the precision of the retrieval,
        # and the recall of the constraints
        # 5. Search Criteria: The search criteria used by the agent
        # 6. Flight Descriptions: The flight descriptions parsed from the example

        correct = grounded and relevant

        eval_dict = dict(
            correct=correct,
            grounded=grounded,
            relevant=relevant,
            question=example['goal'],
            constraints=constraints,
            search_datetime=search_datetime,
            search_criteria=search_criteria,
            flight_descriptions=flight_description_jsons,
            agent_response_data=agent_response_data,
        )

        return eval_dict

    def _parse_evaluator_log(self, evaluator_log_path):
        data = json.load(open(evaluator_log_path))

        goal = None
        datetimes = []
        history = []
        started = False
        for message in data:
            if message.get('observation') == 'browse':
                current_obs = get_obs(message['extras'])
            elif message.get('action') == 'browse_interactive':
                if not started:
                    started = True
                    continue

                # Is agent action message beyond the first step
                timestamp = datetime.fromisoformat(message['timestamp'])
                datetimes.append(timestamp.strftime('%Y-%m-%d %H:%M:%S'))
                step_record = json.loads(message['args']['thought'])

                if step_record.get('obs'):
                    goal = step_record['obs']['goal']

                history.append(
                    (
                        current_obs,
                        step_record['state'],
                        step_record['instruction'],
                        step_record['action'],
                    )
                )

        example = {'goal': goal, 'history': history, 'datetimes': datetimes}
        return example

    def _get_agent_response(self, example):
        last_action = example['history'][-1][-1]
        flight_message = None
        if last_action.startswith('send_msg_to_user'):
            start = len('send_msg_to_user(') + 1
            end = -2
            flight_message = last_action[start:end]
        return flight_message

    def _parse_example_pattern_matches(self, example, field):
        pattern = field_patterns[field]
        matches = []
        for step in example['history']:
            pattern_matches = parse_pattern_matches(
                step[0]['clean_axtree_txt'], pattern
            )
            matches.extend(pattern_matches)
        if len(matches) == 0:
            matches.append(None)
        return matches

    def _parse_flight_description(self, description):
        input_prompt = f"""
Example 3:

Description: {description}

Response:\
"""

        prompt = parse_flight_description_prompt + input_prompt
        for i in range(5):
            completion = self.client.chat.completions.create(
                model='Meta-Llama-3.1-70B-Instruct',
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are an expert at parsing the information in text into JSON format. If there are multiple airlines, join them in one string with "and"',
                    },
                    {'role': 'user', 'content': prompt},
                ],
                temperature=0.1,
                stop=None,
            )
            response = completion.choices[0].message.content

            json_pattern = r'```([^`]+)```'
            match = re.search(json_pattern, response)
            json_text = match.group(1).strip()
            try:
                flight_json = json.loads(json_text)
                return flight_json
            except json.JSONDecodeError as e:
                print(e)

    def _parse_flight_message(self, message):
        input_prompt = f"""
Example 4:

Message: {message}

Response:\
"""

        prompt = parse_flight_message_prompt + input_prompt
        completion = self.client.chat.completions.create(
            model='Meta-Llama-3.1-70B-Instruct',
            messages=[
                {
                    'role': 'system',
                    'content': 'You are an expert at parsing the information in text into Python List with JSON elements. If there are multiple airlines, join them in one string with "and"',
                },
                {'role': 'user', 'content': prompt},
            ],
            temperature=0,
            stop=None,
        )
        response = completion.choices[0].message.content

        python_pattern = r'```python([^`]+)```'
        match = re.search(python_pattern, response)
        python_text = match.group(1).strip()
        python_list = eval(python_text.replace('flights = ', ''))
        return python_list

    def _get_flight_precision(self, candidate, target):
        legitimate_keys = [
            'price',
            'currency',
            'flight_type',
            'airline',
            'departure_city',
            'departure_airport',
            'departure_time',
            'departure_day',
            'departure_date',
            'arrival_city',
            'arrival_airport',
            'arrival_time',
            'arrival_day',
            'arrival_date',
            'total_duration',
        ]
        layover_keys = [
            'layover_city',
            'layover_airport',
            'layover_duration',
            'layover_count',
        ]

        precision = 0
        total = 0
        for key in legitimate_keys:
            if not candidate.get(key):
                # Skip both if key not in candidate or candidate value is None
                continue
            string_contains = isinstance(candidate[key], str) and (
                candidate[key].lower() in str(target.get(key, '')).lower()
                or str(target.get(key, '')).lower() in candidate[key].lower()
            )
            if string_contains or candidate[key] == target.get(key):
                # if candidate[key] in target.get(key):
                precision += 1
            total += 1

        if candidate.get('layovers'):
            for i, cand_layover in enumerate(candidate['layovers']):
                for key in layover_keys:
                    if not cand_layover.get(key):
                        continue
                    if target.get('layovers') and i < len(target['layovers']):
                        tgt_layover = target['layovers'][i]
                        string_contains = isinstance(cand_layover[key], str) and (
                            cand_layover[key].lower()
                            in str(tgt_layover.get(key, '')).lower()
                            or str(tgt_layover.get(key, '')).lower()
                            in cand_layover[key].lower()
                        )
                        if string_contains or cand_layover[key] == tgt_layover.get(key):
                            precision += 1
                    total += 1
        return precision / total

    def _verify_flight_constraint(
        self, search_datetime, search_criteria, top_flights, selected_flight, criterion
    ):
        search_datetime = datetime.strptime(search_datetime, '%Y-%m-%d %H:%M:%S')
        search_datetime = search_datetime.strftime('%a, %b %d, %Y %H:%M:%S')

        input_prompt = f"""
Example 3:

Date and Time: {search_datetime}

Search Criteria:
- Ticket Type: {search_criteria['ticket_type']}
- Seating Class: {search_criteria['seating_class']}
- Origin: {search_criteria['origin']}
- Destination: {search_criteria['destination']}
- Departure: {search_criteria['departure_date']}
- Return: {search_criteria['return_date']}
- Sorting Criterion: {search_criteria['sorting_criterion']}
- Num Passengers: {search_criteria['num_passengers']}

Top flights based on search criteria:
{json.dumps(top_flights, indent=2)}

Selected Flight:
{json.dumps(selected_flight, indent=2)}

Would you say this flight satisfies the following criterion?

Criterion: {criterion}

Respond yes or no. Think step by step. Wrap your responses in the tags <think> </think> and <response> </response>.
"""

        prompt = verify_flight_constraint_prompt + input_prompt
        completion = self.client.chat.completions.create(
            model='Meta-Llama-3.1-70B-Instruct',
            messages=[
                {
                    'role': 'system',
                    'content': """You are an expert at verifying whether a flight satisfies a given criterion. \
During the flight search, the departure city (or origin) will include all airports that serve that area. \
So, if you're searching for flights from New York, for example, the search results might include flights from nearby airports like Newark (EWR) \
or even JFK or LaGuardia. Newark is considered part of the greater New York City area, which is why it shows up in the search results. \
This gives you more flight options that depart from different airports within that region.\
""",
                },
                {'role': 'user', 'content': prompt},
            ],
            temperature=0,
            stop=None,
        )
        response = completion.choices[0].message.content
        ans_dict, success, error = parser(response, ['think', 'response'])

        return ans_dict['response']
