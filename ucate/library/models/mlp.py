import tensorflow as tf
from ucate.library.models import core
from ucate.library.modules import samplers


class BayesianNeuralNetwork(core.BaseModel):
    def __init__(
            self,
            num_examples,
            dim_hidden,
            regression,
            dropout_rate=0.1,
            depth=5,
            *args,
            **kwargs
    ):
        super(BayesianNeuralNetwork, self).__init__(
            *args,
            **kwargs
        )
        self.blocks = []
        print(f"\n*** building a nn with depth of {depth}, where each holds {dim_hidden}-neurons layer with elu activation + dropout ***")
        for i in range(depth):
            self.blocks.append(
                tf.keras.layers.Dense(
                    units=dim_hidden,
                    activation='elu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.5 * (1 / num_examples)),
                    bias_regularizer=tf.keras.regularizers.l2(0.5 * (1 / num_examples))
                )
            )
            self.blocks.append(
                tf.keras.layers.Dropout(rate=0.5 if i == depth - 1 else dropout_rate)
            )
        self.sampler = samplers.NormalSampler(
            dim_output=1,
            num_branches=1,
            num_examples=num_examples,
            beta=0.0
        ) if regression else samplers.BernoulliSampler(
            dim_output=1,
            num_branches=1,
            num_examples=num_examples,
            beta=0.0
        )
        self.regression = regression

    def call(
            self,
            inputs,
            training=None
    ):
        x, y = inputs
        for block in self.blocks:
            x = block(x, training=training)
        py = self.sampler(x)
        self.add_loss(tf.reduce_mean(-py.log_prob(y)))
        mu = py.loc if self.regression else tf.sigmoid(py.logits)
        return mu, py.sample()

    # THIS IS THE REAL MC_STEP USED!!
    def mc_sample_step(
            self,
            inputs
    ):
        x = inputs
        for block in self.blocks:
            x = block(x, training=True) #TODO FALSE FOR NO MC!!
            # Andrew: yes each bloack is a layer, but running it like that is essentially running it through the
            # entire NN. it's broken apart for the cevae, but its running through the entire NN.
            # setting here training=False will eliminate mc!!!!!
        py = self.sampler(x)
        mu = py.loc if self.regression else tf.sigmoid(py.logits)
        # Andrew: every mc step we get a distribution with expectancy (mu) and s.d. (sigma).
        # so we van sample from that distribution, that's py.sample(). Relevant for aleatoric noise, not for cate
        return mu, py.sample()
