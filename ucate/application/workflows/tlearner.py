import os
import json
import shutil
import numpy as np
import tensorflow as tf
from sklearn.utils import resample

from ucate.library import data
from ucate.library import models
from ucate.library import evaluation
from ucate.library.utils import plotting

seed_value = 42
import random
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

def train(
    job_dir,
    dataset_name,
    data_dir,
    trial,
    exclude_population,
    verbose,
    base_filters,
    depth,
    dropout_rate,
    batch_size,
    epochs,
    learning_rate,
    mc_samples,
    bootstrap,
    weighted_bootstrap,
):
    if not(weighted_bootstrap or bootstrap):
        s = "mc-dropout"
    else:
        s = "weighted bootstrap" if weighted_bootstrap else "bootstrap"
    print(f"~~~~~~ Using {s} for uncertainty estimation ~~~~~~")

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    print("TRIAL {:04d} ".format(trial))
    experiment_name = f"bf-{base_filters}_dp-{depth}_dr-{dropout_rate}_bs-{batch_size}_lr-{learning_rate}_ep-{exclude_population}"
    output_dir = os.path.join(
        job_dir,
        dataset_name,
        "tlearner",
        experiment_name,
        f"trial_{trial:03d}",
    )
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    config = {
        "job_dir": job_dir,
        "dataset_name": dataset_name,
        "data_dir": data_dir,
        "exclude_population": exclude_population,
        "trial": trial,
        "base_filters": base_filters,
        "depth": depth,
        "dropout_rate": dropout_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "mc_samples": mc_samples,
    }
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, "w") as fp:
        json.dump(config, fp, indent=4, sort_keys=True)
    # Instantiate data loaders
    dl = data.DATASETS[dataset_name](
        path=data_dir, trial=trial, exclude_population=exclude_population
    )
    if dataset_name in ["acic", "ihdp"]:
        regression = True
        model_name = "mlp"
        loss = tf.keras.losses.MeanSquaredError()
        error = tf.keras.metrics.MeanAbsoluteError()
    else:
        regression = False
        model_name = "cnn"
        loss = tf.keras.losses.BinaryCrossentropy()
        error = tf.keras.metrics.BinaryAccuracy()
    x_train, y_train, t_train, examples_per_treatment = dl.get_training_data()
    idx_0_train = np.where(t_train[:, 0])[0]
    idx_1_train = np.where(t_train[:, 1])[0]

    print(f"*** got training data with examples_per_treatment={examples_per_treatment}. Starting models ***")

    # Instantiate models
    model_0 = models.MODELS[model_name](
        num_examples=examples_per_treatment[0],
        dim_hidden=base_filters,
        dropout_rate=dropout_rate,
        regression=regression,
        depth=depth,
        bootstrap_enabled=bootstrap or weighted_bootstrap,
    )
    model_0.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        # Andrew - this is not the loss actually used. we use log likelihood between our model y and the true y.
        # see "def call" in mlp, the loss is computed there
        metrics=[error],
        loss_weights=[0.0, 0.0],
    )
    model_0_checkpoint = os.path.join(checkpoint_dir, "model_0")
    print(f"*** compiled model 0 with {examples_per_treatment[0]} examples ***")

    model_1 = models.MODELS[model_name](
        num_examples=examples_per_treatment[1],
        dim_hidden=base_filters,
        dropout_rate=dropout_rate,
        regression=regression,
        depth=depth,
        bootstrap_enabled=bootstrap or weighted_bootstrap,
    )
    model_1.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[error],
        loss_weights=[0.0, 0.0],
    )
    model_1_checkpoint = os.path.join(checkpoint_dir, "model_1")
    print(f"*** compiled model 1 with {examples_per_treatment[1]} examples ***")

    #Andrew: this is a model for the propensity!! predict T given X. in t-learner, this is only beeing used to
    # compare against.
    model_prop = models.MODELS[model_name](
        num_examples=sum(examples_per_treatment),
        dim_hidden=base_filters,
        dropout_rate=dropout_rate,
        regression=False,
        depth=2,
        bootstrap_enabled=False,
    )
    model_prop.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.metrics.BinaryAccuracy()],
        loss_weights=[0.0, 0.0],
    )
    model_prop_checkpoint = os.path.join(checkpoint_dir, "model_prop")
    print(f"*** compiled model_prop with {sum(examples_per_treatment)} examples***")

    # Instantiate trainer

    predictions_train_bootstrap = list()
    predictions_test_bootstrap = list()

    for i in range(mc_samples if (bootstrap or weighted_bootstrap) else 1):
        print(f"==== iteration number {i} ====")
        print("\n---- AND NOW - we fit ----")
        if bootstrap:
            ix = [j for j in range(len(x_train))]
            train_ix = resample(ix, replace=True, n_samples=len(x_train))
            x_train_b, y_train_b, t_train_b = x_train[train_ix], y_train[train_ix], t_train[train_ix]
            idx_0_train_b = np.where(t_train_b[:, 0])[0]
            idx_1_train_b = np.where(t_train_b[:, 1])[0]

        sample_weights = np.random.exponential(scale=1.0, size=len(idx_0_train))
        sample_weights /= np.average(sample_weights)
        _ = model_0.fit(
            [x_train[idx_0_train], y_train[idx_0_train]] if not bootstrap else [x_train_b[idx_0_train_b], y_train_b[idx_0_train_b]],  # Andrew: this is x and y. see that training is done on 'inupt'
            [y_train[idx_0_train], np.zeros_like(y_train[idx_0_train])] if not bootstrap else [y_train_b[idx_0_train_b], np.zeros_like(y_train_b[idx_0_train_b])],  # Andrew: this is not used in t-learner
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.3,
            shuffle=True,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=model_0_checkpoint, save_best_only=True, save_weights_only=True
                ),
                tf.keras.callbacks.EarlyStopping(patience=50),
            ],
            verbose=verbose,
            sample_weight=sample_weights if weighted_bootstrap else None
        )

        print("*** fitted model_0 ***")

        sample_weights = np.random.exponential(scale=1.0, size=len(idx_1_train))
        sample_weights /= np.average(sample_weights)
        _ = model_1.fit(
            [x_train[idx_1_train], y_train[idx_1_train]] if not bootstrap else [x_train_b[idx_1_train_b], y_train_b[idx_1_train_b]],
            [y_train[idx_1_train], np.zeros_like(y_train[idx_1_train])] if not bootstrap else [y_train_b[idx_1_train_b], np.zeros_like(y_train_b[idx_1_train_b])],
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.3,
            shuffle=True,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=model_1_checkpoint, save_best_only=True, save_weights_only=True
                ),
                tf.keras.callbacks.EarlyStopping(patience=50),
            ],
            verbose=verbose,
            sample_weight=sample_weights if weighted_bootstrap else None
        )

        print("*** fitted model_1 ***")

        _ = model_prop.fit(
            [x_train, t_train[:, -1]],
            [t_train[:, -1], np.zeros_like(t_train[:, -1])],
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.3,
            shuffle=True,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=model_prop_checkpoint,
                    save_best_only=True,
                    save_weights_only=True,
                ),
                tf.keras.callbacks.EarlyStopping(patience=50),
            ],
            verbose=verbose,
        )

        print("*** fitted model_prop ***")
        print("---- finished fitting on train data ----\n")
        # Restore best models
        model_0.load_weights(model_0_checkpoint)
        model_1.load_weights(model_1_checkpoint)
        model_prop.load_weights(model_prop_checkpoint)

        print("get_prediction calls prediction.mc_sample_tl [with argument of type mlp.BayesianNeuralNetwork.mc_sample "
              "(inherited from core.py)] -> prediction.mc_sample_2 that runs through the model (mlp.mc_sample) for "
              "inference. By doing that - mc_sample_step (implemented in mlp) is called.")

        print("\n^^^^^ evaluation.get_predictions for train ^^^^^")
        # why calling this for train?
        # Andrew: was just interested. Just for the plots, for sanity to see that everything is working
        predictions_train = evaluation.get_predictions(
            dl=dl,
            model_0=model_0,
            model_1=model_1,
            model_prop=model_prop,
            mc_samples=mc_samples,
            test_set=False,
        )
        s = ""
        for k, v in predictions_train.items():
            s += f"{k} with values shape of {v.shape} (type={type(v)}, "
        print(f"** results: {s} **")
        print("^^^^^ evaluation.get_predictions for train is over ^^^^^")

        print("\n^^^^^ evaluation.get_predictions for test ^^^^^")
        predictions_test = evaluation.get_predictions(
            dl=dl,
            model_0=model_0,
            model_1=model_1,
            model_prop=model_prop,
            mc_samples=mc_samples,
            test_set=True,
        )

        s = ""
        for k, v in predictions_test.items():
            s += f"{k} with values shape of {v.shape}, "
        print(f"** results: {s} **")
        a = predictions_test['mu_0']
        print(f"(for example for the first sample - min mc value={np.amin(a[:,0])}, max mc value={np.amax(a[:,0])}, "
              f"avg of mc runs={np.average(a[:,0])}")
        print("^^^^^ evaluation.get_predictions for test is over ^^^^^")

        print(f"if mc is off, all 100 samples should be identical. is that the case? {np.all(a[:,0] == a[:,0][0])}")

        predictions_train_bootstrap.append(predictions_train)
        predictions_test_bootstrap.append(predictions_test)

        # np.savez(os.path.join(output_dir, f"predictions_train_{i}.npz"), **predictions_train)
        # np.savez(os.path.join(output_dir, f"predictions_test_{i}.npz"), **predictions_test)

    print(f"predictions_train_bootstrap is a {type(predictions_train_bootstrap)} in the size of {len(predictions_train_bootstrap)}")

    predictions_train = dict()
    predictions_test = dict()
    if bootstrap or weighted_bootstrap:
        for item in predictions_train_bootstrap:
            for k, v in item.items():
                if k != 'p_t':
                    if k in predictions_train.keys():
                        predictions_train[k] = np.append(predictions_train[k], v[0].reshape([1, *v[0].shape]), axis=0)
                    else:
                        predictions_train[k] = v[0].reshape([1, *v[0].shape])
                else:
                    predictions_train[k] = v

        for item in predictions_test_bootstrap:
            for k, v in item.items():
                if k != 'p_t':
                    if k in predictions_test.keys():
                        predictions_test[k] = np.append(predictions_test[k],v[0].reshape([1, *v[0].shape]), axis=0)
                    else:
                        predictions_test[k] = v[0].reshape([1, *v[0].shape])
                else:
                    predictions_test[k] = v
    else:
        predictions_train = predictions_train_bootstrap[0]
        predictions_test = predictions_test_bootstrap[0]

    print("predictions_train final:")
    s = ""
    for k, v in predictions_train.items():
        s += f"{k} with values shape of {v.shape}, "
    print(f"** results: {s} **")
    print("\nprediction_test final:")
    s = ""
    for k, v in predictions_test.items():
        s += f"{k} with values shape of {v.shape}, "
    print(f"** results: {s} **")

    np.savez(os.path.join(output_dir, "predictions_train.npz"), **predictions_train)
    np.savez(os.path.join(output_dir, "predictions_test.npz"), **predictions_test)

    print(f"\nget_predictions results saved to {os.path.join(output_dir, 'predictions_train.npz')} / _test.npz\n")

    print(f"now creating plot graph to be saved to {os.path.join(output_dir, 'cate_scatter_test.png')}")
    print("data is (predictions_test[mu_1] - predictions_test[mu_0]).mean")

    _, cate = dl.get_test_data(test_set=True)
    check = {"predictions (95% CI)": [(predictions_test["mu_1"] - predictions_test["mu_0"]).mean(0).ravel(),
            cate, 2 * (predictions_test["mu_1"] - predictions_test["mu_0"]).std(0).ravel(),]}

    for k, v in check.items():
        print(f"true cate len: {len(v[0])}")
        print(f"predicted cate len: {len(v[1])}")

    _, cate = dl.get_test_data(test_set=True)
    plotting.error_bars(
        data={
            "predictions (95% CI)": [
                (predictions_test["mu_1"] - predictions_test["mu_0"]).mean(0).ravel(),
                cate,
                2
                * (predictions_test["mu_1"] - predictions_test["mu_0"]).std(0).ravel(),
            ]
        },
        file_name=os.path.join(output_dir, "cate_scatter_test.png"),
    )
    shutil.rmtree(checkpoint_dir)
