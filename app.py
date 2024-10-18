import numpy as np                                                      # Library for numerical computations, used here for array manipulations and loading pre-saved data.
import tensorflow as tf                                                 # TensorFlow, a deep learning library, is used for building neural networks like Encoder, Decoder, and Attention layers.
import streamlit as st                                                  # Streamlit is used to create the web interface for the Spanish-to-English translation app.
import unicodedata                                                      # This module helps normalize and manipulate Unicode characters, especially for cleaning text input.
from tensorflow.keras.preprocessing.sequence import pad_sequences       # Utility to pad input sequences to the same length, ensuring consistent input size for the model.
import re                                                               # Regular Expressions (regex) library, used for text processing to clean and reformat the input sentences.


# Encoder class
class Encoder(tf.keras.Model):
    """
    Encoder model that processes the input sequence into a hidden state vector.

    Attributes:
    ----------
    vocab_size : int
        Size of the input vocabulary.
    emb_dim : int
        Dimensionality of the embedding space.
    enc_units : int
        Number of units in the GRU (hidden size of GRU).
    batch_sz : int
        Batch size for training.
    embedding : tf.keras.layers.Embedding
        Embedding layer that maps each word in the input sequence to a dense vector.
    gru : tf.keras.layers.GRU
        GRU layer that processes the embedded input sequence and returns the hidden state.

    Methods:
    -------
    call(x, hidden)
        Passes the input through the embedding and GRU layers, returning output and hidden state.
    initialize_hidden_state()
        Initializes the hidden state to zeros, used at the start of each batch.
    """

    def __init__(self, vocab_size, emb_dim, enc_units, batch_sz):
        """
        Initializes the Encoder with embedding and GRU layers.

        Parameters:
        ----------
        vocab_size : int
            The size of the vocabulary for the input language.
        emb_dim : int
            The dimensionality of the embedding vectors.
        enc_units : int
            The number of units in the GRU layer.
        batch_sz : int
            The batch size for input sequences.
        """
        super(Encoder, self).__init__()
        self.enc_units = enc_units  # Number of GRU units (hidden size)
        self.batch_sz = batch_sz  # Batch size
        # Embedding layer to convert input sequences into dense vector representations
        self.embedding = tf.keras.layers.Embedding(vocab_size, emb_dim, mask_zero=True)
        # GRU layer to process sequences and return hidden states
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,  # Return all GRU outputs for each time step
                                       return_state=True,  # Also return the final hidden state
                                       recurrent_initializer='glorot_uniform',
                                       # Initialize weights using Glorot uniform
                                       dtype=tf.float32)  # Set dtype to float32

    def call(self, x, hidden):
        """
        Forward pass of the encoder. Processes the input through the embedding and GRU layers.

        Parameters:
        ----------
        x : Tensor
            Input tensor representing the sequence of token IDs (integers).
        hidden : Tensor
            Initial hidden state for the GRU.

        Returns:
        -------
        output : Tensor
            Sequence of outputs from the GRU at each time step.
        state : Tensor
            Final hidden state from the GRU.
        """
        # Ensure the input tensor x is of float32 dtype (default for GRU)
        x = tf.cast(x, dtype=tf.float32)

        # Convert token IDs to embedding vectors
        x = self.embedding(x)

        # Pass the embedded input through the GRU layer
        output, state = self.gru(x, initial_state=hidden)

        return output, state

    def initialize_hidden_state(self):
        """
        Initializes the hidden state for the GRU to zeros.

        Returns:
        -------
        Tensor
            A tensor of zeros with shape (batch_size, units), where `units` is the number of GRU units.
        """
        return tf.zeros((self.batch_sz, self.enc_units), dtype=tf.float32)


# Define the Bahdanau Attention layer class
class BahdanauAttention(tf.keras.layers.Layer):
    """
    Implements Bahdanau (Additive) Attention mechanism.

    Attributes:
    ----------
    W1 : tf.keras.layers.Dense
        A fully connected (Dense) layer to process the query.
    W2 : tf.keras.layers.Dense
        A fully connected (Dense) layer to process the values.
    V : tf.keras.layers.Dense
        A fully connected (Dense) layer to compute the score.

    Methods:
    -------
    call(query, values)
        Computes the context vector and attention weights for a given query and values.
    """

    def __init__(self, units):
        """
        Initializes the Bahdanau Attention layer.

        Parameters:
        ----------
        units : int
            Number of units (neurons) in the dense layers used to compute the attention scores.
        """
        super(BahdanauAttention, self).__init__()
        # Dense layers to transform the query and values for computing the attention score
        self.W1 = tf.keras.layers.Dense(units)  # Fully connected layer for the query
        self.W2 = tf.keras.layers.Dense(units)  # Fully connected layer for the values
        self.V = tf.keras.layers.Dense(1)  # Dense layer to compute the final score

    def call(self, query, values):
        """
        Computes the attention scores and returns the context vector and attention weights.

        Parameters:
        ----------
        query : Tensor
            The query tensor, typically the decoder's hidden state (shape: [batch_size, hidden_size]).
        values : Tensor
            The encoder's output sequence (shape: [batch_size, sequence_length, hidden_size]).

        Returns:
        -------
        context_vector : Tensor
            The weighted sum of the values (shape: [batch_size, hidden_size]).
        attention_weights : Tensor
            The attention weights for each time step (shape: [batch_size, sequence_length, 1]).
        """
        # Cast query and values to float32 to ensure they have the correct dtype
        query = tf.cast(query, dtype=tf.float32)
        values = tf.cast(values, dtype=tf.float32)

        # Expand query to have an extra dimension so it can be added to the values tensor
        query_with_time_axis = tf.expand_dims(query, 1)  # Shape: [batch_size, 1, hidden_size]

        # Compute the attention score: V * tanh(W1 * query + W2 * values)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        # Compute the attention weights using softmax along the sequence dimension (axis=1)
        attention_weights = tf.nn.softmax(score, axis=1)  # Shape: [batch_size, sequence_length, 1]

        # Compute the context vector as the weighted sum of values
        context_vector = attention_weights * values  # Shape: [batch_size, sequence_length, hidden_size]
        context_vector = tf.reduce_sum(context_vector, axis=1)  # Shape: [batch_size, hidden_size]

        return context_vector, attention_weights


# Define the Decoder model
class Decoder(tf.keras.Model):
    """
    Implements the Decoder part of a sequence-to-sequence model with Bahdanau Attention.

    Attributes:
    ----------
    batch_sz : int
        The batch size for training/inference.
    dec_units : int
        The number of GRU units in the decoder.
    attention : BahdanauAttention
        An instance of the BahdanauAttention layer to focus on relevant parts of the input sequence.
    embedding : tf.keras.layers.Embedding
        Embedding layer to convert input words (token IDs) into dense vectors.
    gru : tf.keras.layers.GRU
        GRU (Gated Recurrent Unit) layer for the RNN in the decoder.
    fc : tf.keras.layers.Dense
        Fully connected (Dense) layer to map the GRU output to the target vocabulary size (for predicting the next word).

    Methods:
    -------
    call(x, hidden, enc_output)
        Performs a forward pass through the decoder, returning the predicted token probabilities, the updated hidden state, and attention weights.
    """

    def __init__(self, vocab_size, emb_dim, dec_units, batch_sz):
        """
        Initializes the Decoder with the specified parameters.

        Parameters:
        ----------
        vocab_size : int
            Size of the target vocabulary.
        emb_dim : int
            Dimensionality of the embedding layer.
        dec_units : int
            Number of units in the GRU layer (determines the hidden state size).
        batch_sz : int
            Size of each batch used for training/inference.
        """
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz  # Batch size for training/inference
        self.dec_units = dec_units  # Number of GRU units (decoder units)

        # Attention mechanism using BahdanauAttention
        self.attention = BahdanauAttention(self.dec_units)

        # Embedding layer for converting token IDs into dense vectors
        self.embedding = tf.keras.layers.Embedding(vocab_size, emb_dim)

        # GRU layer (recurrent neural network) with float32 precision for stability
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,  # Return the full sequence (required for attention)
                                       return_state=True,  # Return the final hidden state
                                       recurrent_initializer='glorot_uniform',
                                       dtype=tf.float32)  # Ensure float32 dtype for stability

        # Fully connected (Dense) layer to generate logits for each word in the vocabulary
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden, enc_output):
        """
        Performs a forward pass through the decoder with attention.

        Parameters:
        ----------
        x : Tensor
            The input token (word) to the decoder at the current time step, typically the previously predicted word.
        hidden : Tensor
            The decoder's previous hidden state, used as the query for attention.
        enc_output : Tensor
            The encoder's output sequence, used as the values for attention.

        Returns:
        -------
        x : Tensor
            Logits for the next predicted token, before applying softmax (shape: [batch_size * sequence_length, vocab_size]).
        state : Tensor
            The new hidden state after the GRU (shape: [batch_size, dec_units]).
        attention_weights : Tensor
            Attention weights for each time step in the input sequence (shape: [batch_size, sequence_length, 1]).
        """
        # Ensure the input token is cast to float32
        x = tf.cast(x, dtype=tf.float32)

        # Compute the context vector using attention
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # Pass the input token through the embedding layer
        x = self.embedding(x)

        # Concatenate the context vector with the embedded input token along the last axis
        # The context vector helps the decoder focus on relevant parts of the encoder's output
        x = tf.concat([tf.expand_dims(context_vector, 1), x],
                      axis=-1)  # Shape: [batch_size, 1, embedding_dim + context_vector_dim]

        # Pass the concatenated input (context + word embedding) through the GRU
        output, state = self.gru(x)  # output shape: [batch_size, 1, dec_units], state shape: [batch_size, dec_units]

        # Reshape the output to feed it into the dense layer
        output = tf.reshape(output, (-1, output.shape[2]))  # Shape: [batch_size * 1, dec_units]

        # Pass the GRU output through the dense layer to generate logits for the next word
        x = self.fc(output)  # Shape: [batch_size * 1, vocab_size]

        return x, state, attention_weights


# Function to load pre-trained models and tokenizers
def load_models_and_tokenizers():
    """
    Loads the pre-trained encoder and decoder models, along with the tokenizers for source and target languages.

    The function also loads the word index for both the source and target tokenizers, creates the necessary model
    components (Encoder and Decoder), and restores the model weights from the latest checkpoint.

    Returns:
    -------
    encoder : Encoder
        The loaded encoder model.
    decoder : Decoder
        The loaded decoder model.
    tokenizer_src : tf.keras.preprocessing.text.Tokenizer
        The tokenizer for the source (Spanish) text.
    tokenizer_tgt : tf.keras.preprocessing.text.Tokenizer
        The tokenizer for the target (English) text.
    units : int
        The number of units in the encoder/decoder GRU layers (hidden state size).
    """

    # Initialize empty tokenizers for source and target languages
    tokenizer_src = tf.keras.preprocessing.text.Tokenizer()  # Source language tokenizer (Spanish)
    tokenizer_tgt = tf.keras.preprocessing.text.Tokenizer()  # Target language tokenizer (English)

    # Load the pre-saved word indices for both tokenizers (created during training)
    tokenizer_src.word_index = np.load('tokens/tokenizer_src_word_index.npy', allow_pickle=True).item()
    tokenizer_tgt.word_index = np.load('tokens/tokenizer_trg_word_index.npy', allow_pickle=True).item()

    # Reverse the word index dictionary to get index-to-word mappings for both tokenizers
    tokenizer_src.index_word = {v: k for k, v in tokenizer_src.word_index.items()}
    tokenizer_tgt.index_word = {v: k for k, v in tokenizer_tgt.word_index.items()}

    # Define the vocabulary sizes for the input (source) and target tokenizers
    vocab_inp_size = len(tokenizer_src.word_index)  # Source language vocabulary size
    vocab_tar_size = len(tokenizer_tgt.word_index)  # Target language vocabulary size

    # Set model parameters (embedding dimension and GRU units)
    embedding_dim = 128  # Embedding size (can be adjusted)
    units = 1024  # Number of GRU units (hidden state size)
    batch_size = 64  # Batch size (can be adjusted based on training/inference setup)

    # Create the encoder and decoder models using the loaded vocab sizes and model parameters
    encoder = Encoder(vocab_inp_size, embedding_dim, units, batch_size)  # Instantiate the encoder
    decoder = Decoder(vocab_tar_size, embedding_dim, units, batch_size)  # Instantiate the decoder

    # Define the optimizer (Adam in this case)
    optimizer = tf.keras.optimizers.Adam()

    # Restore the model's weights from the latest checkpoint
    checkpoint_dir = './training_checkpoints'  # Directory where the checkpoints are saved
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

    # Restore the latest checkpoint's weights (for both encoder and decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    # Return the loaded models, tokenizers, and units size
    return encoder, decoder, tokenizer_src, tokenizer_tgt, units


def unicode_to_ascii(s):
    """
    Convert a Unicode string to ASCII by normalizing and removing non-ASCII characters.

    Parameters:
    ----------
    s : str
        The input string containing Unicode characters.

    Returns:
    -------
    str
        A new string that contains only ASCII characters derived from the input.
    """
    # Normalize the input string `s` to its decomposed form (NFD)
    normalized = unicodedata.normalize('NFD', s)

    # Create a new string that includes only ASCII characters
    # Exclude characters with the category "Mark, Nonspacing" (e.g., accents)
    return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')


def preprocess_sentence(text):
    """
    Preprocess the input sentence for translation by normalizing, cleaning,
    and formatting it.

    Parameters:
    ----------
    text : str
        The input sentence to be preprocessed.

    Returns:
    -------
    str
        The preprocessed sentence with added start and end tokens.
    """
    # Normalize the input text to ASCII and convert to lowercase
    text = unicode_to_ascii(text).lower()

    # Replace special characters with a space except for (a-z, A-Z, ".", "?", "!", ",")
    text = re.sub(r'[^a-zA-Z.?!,]', ' ', text)

    # Insert spaces before and after punctuation marks (?, ., !, ,, or ¿)
    text = re.sub(r'([?.!,¿])', r' \1 ', text)

    # Replace one or more consecutive spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove leading and trailing whitespace
    text = text.strip()

    # Add the <start> and <end> tokens to the processed text
    text = '<sos> ' + text + ' <eos>'

    return text



# Translation function
def translate(sentence, encoder, decoder, tokenizer_src, tokenizer_tgt, units, max_length_src, max_length_tgt):
    """
    Translates a given Spanish sentence into English using a trained encoder-decoder model with attention.

    Parameters:
    ----------
    sentence : str
        The input sentence in Spanish to be translated.
    encoder : Encoder
        The trained encoder model.
    decoder : Decoder
        The trained decoder model.
    tokenizer_src : tf.keras.preprocessing.text.Tokenizer
        The tokenizer for the source (Spanish) language.
    tokenizer_tgt : tf.keras.preprocessing.text.Tokenizer
        The tokenizer for the target (English) language.
    units : int
        The number of units in the encoder/decoder GRU layers.
    max_length_src : int
        The maximum length of the input sequences (Spanish).
    max_length_tgt : int
        The maximum length of the output sequences (English).

    Returns:
    -------
    tuple
        A tuple containing:
        - str: The translated sentence in English.
        - str: The original Spanish sentence.
    """

    # Preprocess the input sentence (e.g., removing unwanted characters)
    sentence = preprocess_sentence(sentence)

    # Convert the sentence into a sequence of integers using the source tokenizer
    inputs = [tokenizer_src.word_index[i] for i in sentence.split(' ')]

    # Pad the input sequence to ensure it matches the required input shape
    inputs = pad_sequences([inputs], maxlen=max_length_src, padding='post')

    # Convert the padded input sequence to a tensor with float32 dtype
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)

    # Initialize the hidden state for the encoder
    hidden = [tf.zeros((1, units), dtype=tf.float32)]  # Ensure hidden state uses float32

    # Get the encoder's output and hidden state
    enc_out, enc_hidden = encoder(inputs, hidden)

    # Initialize the decoder's hidden state with the encoder's hidden state
    dec_hidden = enc_hidden

    # Initialize the decoder's input with the start-of-sequence token
    dec_input = tf.expand_dims([tokenizer_tgt.word_index['<sos>']], 0)

    result = ''  # Initialize an empty string to store the translation

    # Loop for the maximum target sequence length
    for t in range(max_length_tgt):
        # Get predictions from the decoder
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_out)

        # Get the predicted ID with the highest probability
        predicted_id = tf.argmax(predictions[0]).numpy()

        # Append the predicted word to the result string
        result += tokenizer_tgt.index_word[predicted_id] + ' '

        # Check for the end-of-sequence token
        if tokenizer_tgt.index_word[predicted_id] == '<eos>':
            return result.strip(), sentence  # Return the result if end token is found

        # Update the decoder input for the next timestep
        dec_input = tf.expand_dims([predicted_id], 0)

    return result.strip(), sentence  # Return the final translation and the original sentence


# Main function for the Streamlit app
def main():
    # Title and initial text
    st.title('Spanish to English Translation')  # App title
    st.write('')  # Blank space for better layout

    # Load models and tokenizers
    encoder, decoder, tokenizer_src, tokenizer_tgt, units = load_models_and_tokenizers()

    # Set maximum lengths based on training data
    max_length_src = 24  # Max length for the source language (Spanish)
    max_length_tgt = 24  # Max length for the target language (English)

    # Input text area for Spanish sentences
    input_text = st.text_area("Enter Spanish text (one sentence per line):")  # Text input for Spanish text

    # Button to trigger translation
    if st.button("Translate"):
        # Split input text by lines (each line is treated as a separate sentence)
        sentences = input_text.strip().split('\n')

        # Create two columns for the original Spanish text and the translated English text
        st1, st2 = st.columns(2)
        st1.header('Spanish')  # Left column header for Spanish text
        st2.header('English')  # Right column header for translated English text
        st.divider()  # Add a divider for neat separation

        # Loop through each sentence entered by the user
        for sentence in sentences:
            # Translate each sentence
            translation = translate(sentence, encoder, decoder, tokenizer_src, tokenizer_tgt, units, max_length_src,
                                    max_length_tgt)

            # Create new columns for each translation result
            st3, st4 = st.columns(2)
            with st3:
                st.subheader(sentence)  # Display original Spanish sentence

            with st4:
                # Display translated English sentence, removing any <eos> token and capitalizing the first letter
                st.subheader(translation[0].removesuffix('<eos>').capitalize().strip())

        st.balloons()  # Celebration balloons effect after translation is complete

# Run the main function if the script is executed directly
if __name__ == '__main__':
    # Configure the Streamlit page with title, icon, and layout
    st.set_page_config(
        page_title='Spanish to English Translator',  # Set the title of the page
        page_icon=':bulb:',  # Set the icon displayed in the tab
        layout='wide'  # Use wide layout for better space utilization
    )

    # CSS to hide the default Streamlit main menu and footer
    hide_style = '''
        <style>
            #MainMenu {visibility: hidden;}  # Hide the main menu
            footer {visibility: hidden;}  # Hide the footer
        </style>
    '''

    # Apply the CSS styles to the Streamlit app
    st.markdown(hide_style, unsafe_allow_html=True)

    # Call the main function to run the app
    main()

