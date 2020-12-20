import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import  Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet

def sigmoid(x):
    '''
    Compute the sigmoid() of the given input
    sigmoid(x) = 1/(1 + exp(-x))
    Input ->  A Real value x
    Output -> sigmoid(x)
    '''
    return 1/(1+np.exp(-x))

def create_class_map(train_csv_path=None):
    '''
    Create a hash table mapping between each integer from 0-79 and each unique class label


    Input ->
    train_csv_path: Path to the "train_curated.csv" file which is used to obtain the list of all class labels

    Output->
    class_map: The class mapping hash table where the key is an integer (0-79) and the value is the corresponding class label
    in English
    '''
    # Read the input csv file
    df_train = pd.read_csv(train_csv_path)

    # Create a list of all unique class labels present in the above file
    LABELS=set()
    all_labels = list(df_train['labels'])
    for row in all_labels:
        for lab in row.split(r','):
            LABELS.add(lab)

    # Sort the list in alphabetical order
    LABELS=list(LABELS)
    LABELS.sort()

    # Create the hash table
    class_map = {}
    for i, label in enumerate(LABELS):
        class_map[i] = label

    return class_map

# Config class is used to share global parameters across various functions
class Config():
    def __init__(self, sampling_rate, n_classes=80):
        self.sampling_rate=sampling_rate
        self.n_classes=n_classes
        self.stft_window_seconds=0.025
        self.stft_hop_seconds=0.010
        self.mel_bands=96
        self.mel_min_hz=20
        self.mel_max_hz=20000
        self.mel_log_offset=0.001
        self.example_window_seconds=1.0
        self.example_hop_seconds=0.5

def clip_to_waveform(clip_path):
    """
    Description -> Decodes a WAV clip into a waveform tensor where the values lie in [-1, +1]
    Inputs->
        clip_path -> Path to .wav file eg. '/test_dir/file1.wav'
    Output -> Tensor corrsponding to the clip with all values between -1 to 1
    """
    clip_data = tf.io.read_file(clip_path)
    waveform, sr = tf.audio.decode_wav(clip_data)
    return tf.squeeze(waveform)

def preprocess(clip_path):
    """
    Description -> Decodes a WAV clip into a batch of log mel spectrum examples
    This function takes the given .wav file, gets it tensor representation, converts it into spectrogram using short-time
    Fourier transform, then converts the spectrogram into log mel spectrogram, finally, it divides it into various windows
    and returns all the windows in a 3-channel format

    Inputs->
        clip_path -> Path to .wav file eg. '/test_dir/file1.wav'

    Output -> Log mel spectrogram windowed features
    """
    # Decode WAV clip into waveform tensor.
    waveform = clip_to_waveform(clip_path)

    # Convert waveform into spectrogram using a Short-Time Fourier Transform.
    # Note that tf.signal.stft() uses a periodic Hann window by default.
    window_length_samples = int(round(config.sampling_rate * config.stft_window_seconds))
    hop_length_samples = int(round(config.sampling_rate * config.stft_hop_seconds))
    fft_length = 2 ** int(np.ceil(np.log2(window_length_samples)))

    magnitude_spectrogram = tf.math.abs(tf.signal.stft(signals=waveform,
                                                       frame_length=window_length_samples,
                                                       frame_step=hop_length_samples,
                                                       fft_length=fft_length))

    # Convert spectrogram into log mel spectrogram.
    num_spectrogram_bins = fft_length // 2 + 1
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=config.mel_bands,
                                                                        num_spectrogram_bins=num_spectrogram_bins,
                                                                        sample_rate=config.sampling_rate,
                                                                        lower_edge_hertz=config.mel_min_hz,
                                                                        upper_edge_hertz=config.mel_max_hz)
    mel_spectrogram = tf.matmul(magnitude_spectrogram, linear_to_mel_weight_matrix)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + config.mel_log_offset)

    # Frame log mel spectrogram into examples.
    spectrogram_sr = 1 / config.stft_hop_seconds
    example_window_length_samples = int(round(spectrogram_sr * config.example_window_seconds))
    example_hop_length_samples = int(round(spectrogram_sr * config.example_hop_seconds))
    features = tf.signal.frame(signal=log_mel_spectrogram,
                               frame_length=example_window_length_samples,
                               frame_step=example_hop_length_samples,
                               pad_end=True,
                               pad_value=0.0,
                               axis=0)

    # Converting mono channel to 3 channels
    features=tf.stack([features,features,features], axis=-1)
    return features

def get_model(weights_path):
    '''
    Return the final Keras model instance with best weights found during training

    Input ->
    weights_path: Path to the best model weights (.h5 file)

    Output ->
    Keras model instance
    '''
    source_model = MobileNet(include_top=False, input_shape=(100, 96, 3))
    x = Flatten()(source_model.layers[-1].output)
    out = Dense(80)(x)
    model = Model(inputs=source_model.input, outputs=out)
    model.load_weights(weights_path)
    return model

# All input clips use a 44.1 kHz sample rate.
SAMPLE_RATE = 44100
config = Config(SAMPLE_RATE)
