from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.eager import context
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine.training import concat, reduce_per_replica

eps = tf.keras.backend.epsilon()
ln = tf.keras.backend.log


# I added ABC and abstractmethod to clarify that this is an abstract class, no one instantiates it.
class BaseModel(tf.keras.Model, ABC):
    def __init__(
            self,
            bootstrap_enabled=False,
            *args,
            **kwargs
    ):
        super(BaseModel, self).__init__(
            *args,
            **kwargs
        )
        self.mc_sample_function = None
        self.bootstrap_enabled = bootstrap_enabled

    @abstractmethod
    def mc_sample_step(
            self,
            inputs
    ):
        raise NotImplementedError('mc_sample step must me impletmented')

    def make_mc_sample_function(self):
        if self.mc_sample_function is not None:
            return self.mc_sample_function

        def predict_function(iterator):
            data = next(iterator)
            data = data if isinstance(data, (list, tuple)) else [data]
            outputs = self.distribute_strategy.run(
                self.mc_sample_step,
                args=data
            )
            outputs = reduce_per_replica(
                outputs,
                self.distribute_strategy,
                reduction='concat'
            )
            return outputs

        self.mc_sample_function = predict_function
        return self.mc_sample_function

    def mc_sample(
            self,
            x,
            batch_size=None,
            steps=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False
    ):
        # Andrew - this is just for doing it in batches & handling distributed (with multiple gpu's). mainly for mnist
        # in practice - mc will just activate dropout in inference
        outputs = None
        with self.distribute_strategy.scope():
            data_handler = data_adapter.DataHandler(
                x=x,
                batch_size=batch_size,
                steps_per_epoch=steps,
                initial_epoch=0,
                epochs=1,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                model=self
            )
            predict_function = self.make_mc_sample_function()
            for _, iterator in data_handler.enumerate_epochs():
                with data_handler.catch_stop_iteration():
                    for step in data_handler.steps():
                        tmp_batch_outputs = predict_function(iterator)
                        if not data_handler.inferred_steps:
                            context.async_wait()
                        batch_outputs = tmp_batch_outputs
                        if outputs is None:
                            outputs = nest.map_structure(
                                lambda batch_output: [batch_output],
                                batch_outputs
                            )
                        else:
                            nest.map_structure_up_to(
                                batch_outputs,
                                lambda output, batch_output: output.append(batch_output),
                                outputs,
                                batch_outputs
                            )
        all_outputs = nest.map_structure_up_to(batch_outputs, concat, outputs)
        return tf_utils.to_numpy_or_python_type(all_outputs)
