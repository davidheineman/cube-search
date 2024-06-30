import numpy as np
import time, warnings, threading, queue

from typing import Optional
from collections import OrderedDict

from .cube_utils import BatchCube, position_permutations, color_permutations, action_permutations, opp_action_permutations


def randomize_input(input_array, rotation_id):
    """
    Randomizes the input, assuming the input has shape (-1, 54, 6)
    """
    pos_perm = position_permutations[rotation_id][np.newaxis, :, np.newaxis]
    col_perm = color_permutations[rotation_id][np.newaxis, np.newaxis]
    input_array = input_array[:, pos_perm, col_perm]

    return input_array

def derandomize_policy(policy, rotation_id):
    """
    Randomizes the policy, assuming the policy has shape (12, )
    """
    return policy[opp_action_permutations[rotation_id]]

def augment_data(inputs, policies, values):
    from cube_utils import position_permutations, color_permutations, action_permutations
        
    inputs = np.array(inputs).reshape((-1, 54, 6))
    sample_size = inputs.shape[0]

    # augement with all color rotations
    sample_idx = np.arange(sample_size)[np.newaxis, :, np.newaxis, np.newaxis]
    pos_perm = position_permutations[:, np.newaxis, :, np.newaxis]
    col_perm = color_permutations[:, np.newaxis, np.newaxis, :]
    inputs = inputs[sample_idx, pos_perm, col_perm]
    inputs = inputs.reshape((-1, 54, 6))

    policies = np.array(policies)
    sample_idx = np.arange(sample_size)[np.newaxis, :, np.newaxis]
    action_perm = action_permutations[:, np.newaxis, :]
    policies = policies[sample_idx, action_perm]
    policies = policies.reshape((-1, 12))
    
    values = np.array(values).reshape((-1, ))
    values = np.tile(values, 48)

    return inputs, policies, values

class Task:
    def __init__(self):
        self.lock = threading.Condition()
        self.input = None
        self.output = None
        self.kill_thread = False

class BaseModel(): 
    """
    The Base Class for my models.  Assuming Keras/Tensorflow backend and
    that the input is a bit array representing a single cube (no history).
    """   
    def __init__(self, use_cache=True, max_cache_size=10000, rotationally_randomize=False, history=1):
        self.learning_rate = .001
        self._model = None # Built and/or loaded later
        self._run_count = 0 # used for measuring computation timing
        self._time_sum = 0 # used for measuring computation timing
        self._cache = None
        self._get_output = None
        self.use_cache = use_cache
        self.rotationally_randomize = rotationally_randomize
        self.max_cache_size = max_cache_size
        self.history = history

        # multithreading, batch evaluation support
        self.multithreaded = False
        self.ideal_batch_size = 128
        self._lock = threading.RLock()
        self._max_batch_size = 1
        self._queue = queue.Queue()
        self._worker_thread = None

        # to reimplement for each model.  Leave off the first dimension.
        self.input_shape = (54, self.history * 6)

    def set_max_batch_size(self, max_batch_size):
        with self._lock:
            self._max_batch_size = max_batch_size
        
        # put a dummy task on the queue to make sure the worker notices the update to the max_batch_size
        dummy_task = Task()
        with dummy_task.lock:
            self._queue.put(dummy_task)

    def get_max_batch_size(self):
        with self._lock:
            return self._max_batch_size

    def _build(self, model):
        """
        The final part of each build method.
        """
        self._model = model
        self._rebuild_function()

    def build(self):
        """
        Build a new neural network using the below architecture
        """
        warnings.warn("'BaseModel.build' should not be used.  The 'build' method should be reimplemented", stacklevel=2)

        model = None 
        self._build(self, model)

    def process_single_input(self, input_array):
        warnings.warn("'BaseModel.process_single_input' should not be used.  The 'process_single_input' method should be reimplemented", stacklevel=2)
        input_array = input_array.reshape((self.history, 54, 6))
        if self.history > 1:
            input_array = np.rollaxis(input_array,  1, 0)
            input_array = input_array.reshape((1, 54, self.history * 6))
        return input_array

    def _rebuild_function(self):
        """
        Rebuilds the function associated with this network.  This is called whenever
        the network is changed.
        """
        from keras import backend as K
        self._cache = OrderedDict()

        # run model once to make sure it loads correctly (needed for K.function to work on new models)
        trivial_input = np.zeros((1, ) + self.input_shape)
        self._model.predict(trivial_input)

        self._get_output = K.function([self._model.input, K.learning_phase()], [self._model.output[0], self._model.output[1]])
        
        if self.multithreaded:
            if self._worker_thread is not None:
                self.stop_worker_thread()
            self.start_worker_thread()

    def _raw_function(self, input_array):
        t1 = time.time()
        #return self._model.predict(input_array)
        out = self._get_output([input_array, 0])
        self._run_count += 1
        self._time_sum = time.time() - t1 
        return out

    def _raw_function_worker(self):
        import numpy as np
        task_list = []

        while True:
            # retrieve items from the queue
            task = self._queue.get()
            with task.lock:
                if task.kill_thread:
                    task.lock.notify()
                    return
                
                if task.input is not None: #ignore other tasks as dummy tasks
                    task_list.append(task)

            if task_list and len(task_list) >= min(self.ideal_batch_size, self.get_max_batch_size()):
                array = np.array([task.input.squeeze(axis=0) for task in task_list])
                policies, values = self._get_output([array, 0])

                for p, v, task in zip(policies, values, task_list):
                    with task.lock:
                        task.output = [p[np.newaxis], v[np.newaxis]]
                        task.lock.notify() # mark as being complete

                task_list = []

    def start_worker_thread(self):
        self._worker_thread = threading.Thread(target=self._raw_function_worker, args=())
        self._worker_thread.daemon = True
        self._worker_thread.start()

    def stop_worker_thread(self):
        poison_pill = Task()
        with poison_pill.lock:
            poison_pill.kill_thread = True
            self._queue.put(poison_pill) # put task on queue to be processed
            poison_pill.lock.wait() # wait for poison pill to be processed
        self._worker_thread.join() # wait until thread finishes
        self._worker_thread = None

    def _raw_function_pass_to_worker(self, input_array):
        task = Task()

        # put the value on the queue to be processed
        with task.lock:
            task.input = input_array
            self._queue.put(task) # put task on queue to be processed
            task.lock.wait() # wait until task is processed
            return task.output # return output

    def _inner_function(self, input_array):
        """
        The function which computes the output to the array.
        Assume input_array has shape (-1, 56, 4) where -1 represents the history.
        """ 
        if self.use_cache:
            key = input_array.tobytes()
            if key in self._cache:
                self._cache.move_to_end(key, last=True)
                return self._cache[key]
        
        input_array = self.process_single_input(input_array)
        if self.multithreaded:
            policy, value = self._raw_function_pass_to_worker(input_array)
        else:
            policy, value = self._raw_function(input_array)
        
        policy = policy.reshape((12,))
        value = value[0, 0]

        if self.use_cache:
            self._cache[key] = (policy, value)
            if len(self._cache) > self.max_cache_size:
                self._cache.popitem(last=False)

        return policy, value

    def function(self, input_array):
        """
        The function which computes the output to the array.
        If self.rotationally_randomize is true, will first randomly rotate input
        and (un-)rotate corresponding policy output.
        Assume input_array has shape (-1, 56, 4) where -1 represents the history.
        """ 
        if self.rotationally_randomize:
            rotation_id = np.random.choice(48)
            input_array = randomize_input(input_array, rotation_id)

        policy, value = self._inner_function(input_array)

        if self.rotationally_randomize:
            policy = derandomize_policy(policy, rotation_id)

        return policy, value

    def load_from_file(self, path):
        self._model.load_weights(path)
        self._rebuild_function()

    def save_to_file(self, path):
        self._model.save_weights(path)
    
    def train_on_data(self, data):
        """
        data: list of inputs, policies, values as arrays (assume already preprocessed for training)
        """

        inputs, outputs_policy, outputs_value = data
        self._model.fit(x=inputs, 
                        y={'policy_output': outputs_policy, 'value_output': outputs_value}, 
                        epochs=1, verbose=0)
        print("AAA done training")
        self._rebuild_function()

    @staticmethod
    def augment_data(inputs, policies, values):
        """
        Augment data with all 48 color rotations
        """
        from cube_utils import position_permutations, color_permutations, action_permutations

        inputs = np.array(inputs).reshape((-1, 54, 6))
        sample_size = inputs.shape[0]

        sample_idx = np.arange(sample_size)[np.newaxis, :, np.newaxis, np.newaxis]
        pos_perm = position_permutations[:, np.newaxis, :, np.newaxis]
        col_perm = color_permutations[:, np.newaxis, np.newaxis, :]
        inputs = inputs[sample_idx, pos_perm, col_perm]
        inputs = inputs.reshape((-1, 54, 6))

        policies = np.array(policies)
        sample_idx = np.arange(sample_size)[np.newaxis, :, np.newaxis]
        action_perm = action_permutations[:, np.newaxis, :]
        policies = policies[sample_idx, action_perm]
        policies = policies.reshape((-1, 12))
        
        values = np.array(values).reshape((-1, ))
        values = np.tile(values, 48)

        return inputs, policies, values

    @staticmethod
    def validate_data(inputs, policies, values, gamma=.95):
        """
        Validate the input, policy, value data to make sure it is of good quality.
        It must be in order and not shuffled.
        """
        from cube_utils import BatchCube
        import math

        next_state = None
        next_value = None

        for state, policy, value in zip(inputs, policies, values):
            cube = BatchCube()
            cube.load_bit_array(state)
            
            if next_state is not None:
                assert next_state.shape == state.shape
                assert np.array_equal(next_state, state), "\nstate:\n" + str(state) + "\nnext_state:\n" + str(next_state)
            if next_value is not None:
                assert round(math.log(next_value, .95)) == round(math.log(value, .95)), "next_value:" + str(next_value) + "   value:" + str(value)

            action = np.argmax(policy)
            cube.step([action])

            if value == 0 or value == gamma:
                next_value = None
                next_state = None
            else:
                next_value = value / gamma
                next_state = cube.bit_array().reshape((54, 6))

    @staticmethod
    def preprocess_training_data(inputs, policies, values):
        """
        Convert training data to arrays in preparation for saving.
        
        Don't augment, since this takes up too much space and doesn't cost much in time to
        do it later.

        Also keep the inputs shape as (-1, 54, 6) so that other models can use the same
        data.  Similarly, assume the inputs are stored only with the last state.
        """
        
        # convert to arrays
        inputs = np.array(inputs).reshape((-1, 54, 6))
        policies = np.array(policies)
        values = np.array(values).reshape((-1, ))

        return inputs, policies, values

    def process_training_data(self, inputs, policies, values, augment=True):
        """
        Convert training data to arrays.  
        Augment data
        Reshape to fit model input.
        """
        warnings.warn("'BaseModel.process_training_data' should not be used.  The 'process_single_input' method should be reimplemented", stacklevel=2)
        # augment with all 48 color rotations
        if augment:
            inputs, policies, values = augment_data(inputs, policies, values)

        # process arrays now to save time during training
        if self.history == 1:
            inputs = inputs.reshape((-1, 54, 6))
        else:
            # use that the inputs are in order to attach the history
            # use the policy/input match to determine when we reached a new game
            next_cube = None
            input_array_with_history = None
            input_list = []
            for state, policy in zip(inputs, policies):
                cube = BatchCube()
                cube.load_bit_array(state)
                
                if next_cube is None or cube != next_cube:
                    # blank history
                    input_array_history = np.zeros((self.history-1, 54, 6), dtype=bool)
                else:
                    input_array_history = input_array_with_history[:-1]
                
                input_array_state = state.reshape((1, 54, 6))
                input_array_with_history = np.concatenate([input_array_state, input_array_history], axis=0)
                
                input_array = np.rollaxis(input_array_with_history,  1, 0)
                input_array = input_array.reshape((54, self.history * 6))
                input_list.append(input_array)
                
                action = np.argmax(policy)
                next_cube = cube.copy()
                next_cube.step([action])
                
            inputs = np.array(input_list)

        return inputs, policies, values




class CubeModel:
    _model = None  # type: Optional[models.BaseModel]

    def __init__(self):
        pass

    def load_from_config(self, filepath: Optional[str] = None):
        """
        Build a model from the config file settings.
        :param filepath: Optional string to filepath of model weights.
                         If None (default) then it will load based on the config file.
        """
        import config

        if filepath is None:
            assert False, "Fill in this branch"

        self.load(config.model_type, filepath, **config.model_kwargs)

    def load(self, model_type: str, filepath: str, **kwargs):
        """
        Build a model.
        :param model_type: The name of the model class in models.py
        :param filepath: The path to the model weights.
        :param kwargs: Key word arguements for initializing the model class (the one given by model_type).
        """
        model_constructor = models.__dict__[model_type]  # get model class by name
        self._model = model_constructor(**kwargs)

        self._model.build()
        self._model.load_from_file(filepath)

    def _function(self):
        assert (self._model is not None), "No model loaded"
        return self._model.function
