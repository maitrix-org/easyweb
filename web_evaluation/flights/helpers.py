import contextlib
import re

import joblib
from browsergym.utils.obs import flatten_axtree_to_str
from utils import (
    ParseError,
    parse_html_tags_raise,
)

# dict_keys(['url', 'screenshot', 'status_code', 'error', 'open_pages_urls', 'active_page_index', 'dom_object', 'axtree_object', 'extra_element_properties', 'last_browser_action', 'last_browser_action_error', 'focused_element_bid', 'scroll_position'])


def get_obs(last_obs):
    cur_axtree_txt = ''
    error_prefix = ''
    current_obs = {}

    # The browser output observation belongs to OpenDevin
    if last_obs.get('error'):
        # add error recovery prompt prefix
        error_prefix = f'IMPORTANT! Last action is incorrect:\n{last_obs.get("last_browser_action")}\n{last_obs.get("last_browser_action_error")}\nThink again with the current observation of the page.\n'
    try:
        cur_axtree_txt = flatten_axtree_to_str(
            last_obs['axtree_object'],
            extra_properties=last_obs['extra_element_properties'],
            with_clickable=True,
            filter_visible_only=True,
        )
        scroll_progress = (
            1
            - last_obs['scroll_position']['remainingPixels']
            / last_obs['scroll_position']['documentHeight']
        )
        cur_axtree_txt = (
            f"URL {last_obs['url']}\n"
            f"Scroll Position: {last_obs['scroll_position']['scrollTop']}, "
            f"Window Height: {last_obs['scroll_position']['windowHeight']}, "
            f"Webpage Height: {last_obs['scroll_position']['documentHeight']}, "
            f"Remaining Pixels: {last_obs['scroll_position']['remainingPixels']}, "
            f"Scrolling Progress: {scroll_progress:.1%}\n"
        ) + cur_axtree_txt
    except Exception as e:
        print('Error when trying to process the accessibility tree: %s', e)

    clean_axtree_lines = []
    num_static_text_lines = 0
    max_static_text_lines = 20
    last_bracket_line = 0
    max_after_last_bracket_lines = 10
    for i, line in enumerate(cur_axtree_txt.split('\n')):
        if line.strip().startswith('['):
            last_bracket_line = i

    for i, line in enumerate(cur_axtree_txt.split('\n')):
        if line.strip().startswith('StaticText') or line.strip().startswith(
            'ListMarker'
        ):
            num_static_text_lines += 1
        else:
            num_static_text_lines = 0

        if num_static_text_lines <= max_static_text_lines and i < (
            last_bracket_line + max_after_last_bracket_lines
        ):
            clean_axtree_lines.append(line)

    clean_axtree_txt = '\n'.join(clean_axtree_lines)

    obs_prompt = clean_axtree_txt
    if len(error_prefix) > 0:
        obs_prompt = f'{error_prefix}\n' + obs_prompt

    current_obs = {
        'clean_axtree_txt': obs_prompt,
        'raw_axtree_txt': cur_axtree_txt,
        # 'axtree_txt': "AXSTART "+cur_axtree_txt+" AXEND",
        'error_prefix': error_prefix,
    }
    return current_obs


flight_description_pattern = r'\[\d+\] link [\"|\'](.*?)Select flight[\"|\']'
ticket_type_pattern = r'\[\d+\] combobox [\"|\']Change ticket type. (.*?)[\"|\'], live=[\"|\']polite[\"|\']'
seating_class_pattern = r'\[\d+\] combobox [\"|\']Change seating class. (.*?)[\"|\'], live=[\"|\']polite[\"|\']'
origin_pattern = (
    r'\[\d+\] combobox [\"|\']Where from\?.*[\"|\'] value=[\"|\'](.+)[\"|\'], clickable'
)
destination_pattern = (
    r'\[\d+\] combobox [\"|\']Where to\?.*[\"|\'] value=[\"|\'](.+)[\"|\'], clickable'
)
departure_date_pattern = (
    r'\[\d+\] textbox [\"|\']Departure[\"|\'] value=[\"|\'](.*?)[\"|\'], clickable'
)
return_date_pattern = (
    r'\[\d+\] textbox [\"|\']Return[\"|\'] value=[\"|\'](.*?)[\"|\'], clickable'
)
sorting_criterion_pattern = r'\[\d+\] button [\"|\'](.*?), Change sort order.[\"|\'], hasPopup=[\"|\']menu[\"|\']'
num_passengers_pattern = (
    r'\[\d+\] button [\"|\'](.*?), change number of passengers.[\"|\']'
)

field_patterns = {
    'flight_description': flight_description_pattern,
    'ticket_type': ticket_type_pattern,
    'seating_class': seating_class_pattern,
    'origin': origin_pattern,
    'destination': destination_pattern,
    'departure_date': departure_date_pattern,
    'return_date': return_date_pattern,
    'sorting_criterion': sorting_criterion_pattern,
    'num_passengers': num_passengers_pattern,
}


def parse_pattern_matches(axtree_txt, pattern):
    matches = []
    for line in axtree_txt.split('\n'):
        match = re.search(pattern, line)
        if match:
            matches.append(match.group(1))
    return matches


def parser(text, output_keys):
    try:
        ans_dict = parse_html_tags_raise(
            text,
            keys=output_keys,
            merge_multiple=True,
        )
    except ParseError as e:
        return None, False, str(e)
    return ans_dict, True, ''


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
