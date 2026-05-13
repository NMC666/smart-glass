import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

import math
from typing import Tuple, List
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import mltk.core as mltk_core
from mltk.core.preprocess.audio.audio_feature_generator import AudioFeatureGeneratorSettings
from mltk.core.preprocess.utils import tf_dataset as tf_dataset_utils
from mltk.core.preprocess.utils import audio as audio_utils
from mltk.core.preprocess.utils import split_file_list, shuffle_file_list_by_group
from mltk.models.shared import tenet
DATASET_DIR = os.path.abspath("prepared_heysnips")
class MyModel(
    mltk_core.MltkModel,
    mltk_core.TrainMixin,
    mltk_core.DatasetMixin,
    mltk_core.EvaluateClassifierMixin,
):
    pass
my_model = MyModel()
my_model.version = 1
my_model.description = "Keyword spotting classifier to detect: hey_snips"
my_model.epochs = 25
my_model.batch_size = 512
my_model.checkpoint["monitor"] = "val_accuracy"
my_model.train_callbacks = [
    tf.keras.callbacks.TerminateOnNaN(),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=12,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1,
    ),
]
my_model.tflite_converter["optimizations"] = [tf.lite.Optimize.DEFAULT]
my_model.tflite_converter["supported_ops"] = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
my_model.tflite_converter["inference_input_type"] = np.int8
my_model.tflite_converter["inference_output_type"] = np.int8
my_model.tflite_converter["representative_dataset"] = "generate"
frontend_settings = AudioFeatureGeneratorSettings()
frontend_settings.sample_rate_hz = 16000
frontend_settings.sample_length_ms = 1500
frontend_settings.window_size_ms = 30
frontend_settings.window_step_ms = 10
frontend_settings.filterbank_n_channels = 40
frontend_settings.filterbank_lower_band_limit = 125.0
frontend_settings.filterbank_upper_band_limit = 7500.0
frontend_settings.noise_reduction_enable = True
frontend_settings.noise_reduction_smoothing_bits = 10
frontend_settings.noise_reduction_even_smoothing = 0.025
frontend_settings.noise_reduction_odd_smoothing = 0.06
frontend_settings.noise_reduction_min_signal_remaining = 0.40
frontend_settings.dc_notch_filter_enable = True
frontend_settings.dc_notch_filter_coefficient = 0.95
frontend_settings.quantize_dynamic_scale_enable = True
frontend_settings.quantize_dynamic_scale_range_db = 40.0
my_model.model_parameters.update(frontend_settings)
my_model.input_shape = (
    frontend_settings.spectrogram_shape[0],
    1,
    frontend_settings.spectrogram_shape[1],
)
my_model.classes = [
    "hey_snips",
    "_unknown_",
    "_silence_"
]
my_model.class_weights = "balanced"
validation_split = 0.15
def my_model_builder(model: MyModel) -> tf.keras.Model:
    keras_model = tenet.TENet12(
        input_shape=model.input_shape,
        classes=model.n_classes,
        channels=40,
        blocks=6,
    )
    keras_model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
            epsilon=1e-8,
        ),
        metrics=["accuracy"],
    )
    return keras_model
my_model.build_model_function = my_model_builder
my_model.keras_custom_objects["MultiScaleTemporalConvolution"] = (
    tenet.MultiScaleTemporalConvolution
)

def audio_pipeline(
    path_batch: np.ndarray,
    label_batch: np.ndarray,
    seed: np.ndarray,
) -> np.ndarray:
    batch_length = path_batch.shape[0]
    height, width = frontend_settings.spectrogram_shape
    x_shape = (batch_length, height, 1, width)
    x_batch = np.empty(x_shape, dtype=np.int8)

    out_length = int(
        frontend_settings.sample_rate_hz
        * frontend_settings.sample_length_ms
        / 1000
    )

    # Label mapping:
    # 0 = hey_snips
    # 1 = unknown
    # 2 = silence

    for i, audio_path in enumerate(path_batch):
        if isinstance(audio_path, bytes):
            audio_path = audio_path.decode("utf-8")

        try:
            sample, original_sample_rate = audio_utils.read_audio_file(
                audio_path,
                return_numpy=True,
                return_sample_rate=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to read audio file: {audio_path}, err={e}")

        if original_sample_rate != frontend_settings.sample_rate_hz:
            sample = audio_utils.resample(
                sample,
                orig_sr=original_sample_rate,
                target_sr=frontend_settings.sample_rate_hz,
            )

        sample = sample.astype(np.float32)

        if len(sample.shape) > 1:
            sample = sample[:, 0]

        sample = np.nan_to_num(sample)

        max_abs = np.max(np.abs(sample)) if sample.size > 0 else 0.0
        if max_abs > 2.0:
            sample = sample / 32768.0

        sample = audio_utils.adjust_length(
            sample,
            out_length=out_length,
            trim_threshold_db=30,
            offset=0.5,
        )

        sample = np.clip(sample, -1.0, 1.0)

        if len(sample) < out_length:
            sample = np.pad(sample, (0, out_length - len(sample)), mode="constant")
        elif len(sample) > out_length:
            sample = sample[:out_length]

        spectrogram = audio_utils.apply_frontend(
            sample=sample,
            settings=frontend_settings,
            dtype=np.int8,
        )

        spectrogram = np.expand_dims(spectrogram, axis=-2)
        x_batch[i] = spectrogram

    return x_batch

class MyDataset(mltk_core.MltkDataset):
    def __init__(self):
        super().__init__()
        self.pools = []
        self.summary = ""
    def summarize_dataset(self) -> str:
        s = self.summary
        s += mltk_core.MltkDataset.summarize_class_counts(my_model.class_counts)
        return s
    def load_dataset(
        self,
        subset: str,
        test: bool = False,
        **kwargs,
    ):
        if subset == "training":
            train_ds = self.load_subset("training", test=test)
            val_ds = self.load_subset("validation", test=test)
            return train_ds,None, val_ds
        elif subset in ("validation", "evaluation"):
            return self.load_subset("validation", test=test)
        else:
            return self.load_subset(subset, test=test)
    def unload_dataset(self):
        for pool in self.pools:
            pool.shutdown()
        self.pools.clear()
    def load_subset(self, subset: str, test: bool) -> tf.data.Dataset:
        if not os.path.isdir(DATASET_DIR):
            raise RuntimeError(f"Dataset directory not found: {DATASET_DIR}")
        if subset in ("validation", "evaluation"):
            split = (0, validation_split)
        elif subset == "training":
            split = (validation_split, 1)
        else:
            split = None
        max_samples_per_class = my_model.batch_size if test else -1
        if subset:
            class_counts = my_model.class_counts[subset]
        else:
            class_counts = my_model.class_counts
        features_ds, labels_ds = tf_dataset_utils.load_audio_directory(
            directory=DATASET_DIR,
            classes=my_model.classes,
            onehot_encode=False,
            shuffle=True,
            seed=42,
            max_samples_per_class=max_samples_per_class,
            split=split,
            return_audio_data=False,  
            class_counts=class_counts,
            list_valid_filenames_in_directory_function=self.list_valid_filenames_in_directory,
        )
        if subset:
            try:
                seed_counter = tf.data.Dataset.counter()
            except Exception:
                seed_counter = tf.data.experimental.Counter()
            features_ds = features_ds.zip((features_ds, labels_ds, seed_counter))
            per_job_batch_size = my_model.batch_size
            features_ds = features_ds.batch(
                per_job_batch_size,
                drop_remainder=True,
            )
            labels_ds = labels_ds.batch(
                per_job_batch_size,
                drop_remainder=True,
            )
            features_ds, pool = tf_dataset_utils.parallel_process(
                features_ds,
                audio_pipeline,
                dtype=np.int8,
                n_jobs=4,
                name=subset,
            )
            expected_feature_shape = (
                frontend_settings.spectrogram_shape[0],
                1,
                frontend_settings.spectrogram_shape[1],
            )
            expected_label_shape = ()
            def fix_shapes(x, y):
                x = tf.ensure_shape(x, expected_feature_shape)
                y = tf.ensure_shape(y, expected_label_shape)
                return x, y
            self.pools.append(pool)
            features_ds = features_ds.unbatch()
            labels_ds = labels_ds.unbatch()
        ds = tf.data.Dataset.zip((features_ds, labels_ds))
        expected_feature_shape = (
            frontend_settings.spectrogram_shape[0],
            1,
            frontend_settings.spectrogram_shape[1],
        )
        expected_label_shape = ()
        def fix_shapes(x, y):
            x = tf.ensure_shape(x, expected_feature_shape)
            y = tf.ensure_shape(y, expected_label_shape)
            return x, y
        ds = ds.map(
            fix_shapes,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.cache()
        if not test:
            ds = ds.shuffle(
                buffer_size=2048,
                reshuffle_each_iteration=True,
            )
        ds = ds.batch(my_model.batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
    def list_valid_filenames_in_directory(
        self,
        base_directory: str,
        search_class: str,
        white_list_formats: List[str],
        split: Tuple[float, float],
        follow_links: bool,
        shuffle_index_directory: str,
    ) -> Tuple[str, List[str]]:
        class_dir = os.path.join(base_directory, search_class)
        if not os.path.isdir(class_dir):
            raise RuntimeError(f"Class directory not found: {class_dir}")
        file_list = []
        for root, _, files in os.walk(class_dir, followlinks=follow_links):
            for fname in files:
                if not fname.lower().endswith(tuple(white_list_formats)):
                    continue
                abs_path = os.path.join(root, fname)
                if os.path.getsize(abs_path) == 0:
                    continue
                rel_path = os.path.relpath(abs_path, base_directory)
                file_list.append(rel_path.replace("\\", "/"))
        if len(file_list) == 0:
            raise RuntimeError(f"No samples found for class: {search_class}")
        rng = np.random.RandomState(42)
        file_list = sorted(file_list)
        rng.shuffle(file_list)
        if split is None:
            return search_class, file_list
        n_files = len(file_list)
        start = int(math.floor(split[0] * n_files))
        stop = int(math.ceil(split[1] * n_files))
        filenames = file_list[start:stop]
        return search_class, filenames
my_model.dataset = MyDataset()
my_model.model_parameters["average_window_duration_ms"] = 600
my_model.model_parameters["detection_threshold_list"] = [
    int(0.85 * 255),
    int(1.00 * 255),  # _unknown_
    int(1.00 * 255),  # _silence_
]
my_model.model_parameters["suppression_ms"] = 1500
my_model.model_parameters["minimum_count"] = 3
my_model.model_parameters["volume_gain"] = 0.0
my_model.model_parameters["latency_ms"] = 10
my_model.model_parameters["verbose_model_output_logs"] = False
if __name__ == "__main__":
    from mltk import cli
    cli.get_logger(verbose=True)
    test_mode_enabled = False   
    print("Input shape:", my_model.input_shape)
    print("Classes:", my_model.classes)
    train_results = mltk_core.train_model(
        my_model,
        clean=True,
        test=test_mode_enabled,
    )
    print(train_results)
    eval_results = mltk_core.evaluate_model(
        my_model,
        verbose=True,
        test=test_mode_enabled,
    )
    print(eval_results)
    profile_results = mltk_core.profile_model(
        my_model,
        test=test_mode_enabled,
    )
    print(profile_results)
