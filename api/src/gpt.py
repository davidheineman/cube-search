import os, datetime, time
import openai
from openai import OpenAI
from collections import Counter

import numpy as np
from .model import BaseModel
from .cube_utils import BatchCube, BASIC_MOVES


CUBE_PROMPT = """You are solving a Rubriks cube. Here is the current state:
{cube_state}
Select the next position from the following options: {options}. Make sure to think step by step about your response, then return the next position at the end (e.g., My next move is NEXT_MOVE)."""


DEFAULT_KWARGS_OPENAI = {
    'top_p': 0.9,
    'temperature': 0.9,
    'max_new_tokens': 256, 
    'output_scores': False
}


class CubeGPT(BaseModel): 
    """ GPT wrapper to explain, then propose, the next move """   

    def __init__(self, use_cache=True, max_cache_size=10000, rotationally_randomize=False, history=1):
        BaseModel.__init__(self, use_cache, max_cache_size, rotationally_randomize, history)
        self.input_shape = (6 * 6, 3, 3)

        self.openai_model, self.openai_client = self.openai_init('.OPENAI_SECRET')
    
    def _inner_function(self, input_array):
        """
        The function which computes the output to the array.
        Assume input_array has shape (-1, 56, 4) where -1 represents the history.
        """
        current_state = input_array[0, :, :]

        cube = BatchCube()
        cube.load_bit_array(current_state)
        
        print(f'\n{cube}')
        
        # Porbabilities of taking each of the 12 actions on the cube
        policy = np.array([0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.1, 0.1, 0.05, 0.1, 0.1, 0.1])

        # Value of the current state towards the goal (0-1)
        value = np.array([[0.8]])

        return policy, value

    def _function(self):
        """ Provides the model_policy_value() function used by MCTSAgent to propose moves """
        return self._inner_function

    def openai_init(self, secret_path, model_name="gpt-3.5-turbo"):
        openai_model = model_name
        if not os.path.exists(secret_path): 
            raise RuntimeError(f'Need an OpenAI Key! Did not find a key at {secret_path}')
        with open(secret_path, 'r') as f: 
            api_key = f.read().strip()
        openai_client = OpenAI(api_key=api_key)
        return openai_model, openai_client

    def _call_openai(self, p, params, max_retries=7, base_delay=2):
        retries = 0
        while retries < max_retries:
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[{ "role": "user", "content": p }],
                    temperature=params['temperature'],
                    max_tokens=params['max_new_tokens'],
                    top_p=params['top_p'],
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stop=None
                )
                return response
            except (openai.RateLimitError, openai.InternalServerError) as e:
                retries += 1
                print(f"OpenAI API request exceeded rate limit: {e}. Retrying ({retries}/{max_retries})...")
                if retries < max_retries:
                    delay = base_delay * (2 ** retries)
                    time.sleep(delay)
                else:
                    raise RuntimeError('OpenAI API failed to respond')

    def generate_gpt(self, prompt, concurrent=False, **kwargs):
        """ Generate with GPT. """
        start_time = datetime.datetime.now()
        if not isinstance(prompt, list): prompt = [prompt]
        
        params = DEFAULT_KWARGS_OPENAI.copy()
        for k, kwarg in kwargs.items():
            if k in params.keys():
                params[k] = kwarg

        # print(f'Generating {len(prompt)} examples on {self.openai_model} with params {params}')
        
        if concurrent:
            # Query OpenAI using threading
            with concurrent.futures.ThreadPoolExecutor() as exec:
                futures = [exec.submit(self._call_openai, p, params) for p in prompt]
                cands = [f.result() for f in concurrent.futures.as_completed(futures)]
            cands = [c.choices[0].message.content for c in cands]
        else:
            # Query OpenAI sequentially
            cands = []
            for p in prompt:
                resp = self._call_openai(p, params)
                cands += [resp.choices[0].message.content]

        duration = (datetime.datetime.now() - start_time).total_seconds()

        # print(f"Generated {len(prompt)} queries in {duration:.2f}s at {len(prompt)/duration:.2f} prompt/s.")

        return cands
    
    def parse_response(self, response):
        """ Convert ['My next move is U.', 'My next move is U.', ...] -> [U, U, ...] """
        parsed = []
        for resp in response:
            words = resp.split()
            move = words[-1].strip('.')
            if move not in BASIC_MOVES:
                print(f'Could not parse "{resp}". Skipping...')
                continue
            parsed += [move]
        return parsed
    
    def next_move(self, cube, num_chains=1):
        formatted_moves = ', '.join(BASIC_MOVES)
        formatted_prompt = CUBE_PROMPT.format(cube_state=str(cube), options=formatted_moves)

        print(formatted_prompt)

        response = self.generate_gpt([formatted_prompt for _ in range(num_chains)])

        response = self.parse_response(response)

        move_count = dict(Counter(response))
        return move_count
