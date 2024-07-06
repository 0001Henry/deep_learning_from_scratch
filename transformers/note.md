# Attention Mechanism
- enable models to focus selectively on parts of the input data that are most relevant to the task at hand
- dynamically assigns importance to different parts of the data based on the content, enhancing the model's ability to make context-aware decisions

## Core Concepts
- **Self-Attention**: This allows each element of the input sequence to evaluate and assign a weight to every other element, which is crucial for understanding the relationships within the sequence.
    1. **Query, Key, and Value Vectors**: Each element of the input is transformed into three vectors - Query (Q), Key (K), and Value (V).
    2. **Score Calculation**: The score is calculated by taking the dot product of the Query with all Keys.
       $$
       \text{Score} = QK^T
       $$
    3. **Softmax to Get Weights**: The scores are then passed through a softmax layer to normalize them into a probability distribution.
       $$
       \text{Weights} = \text{softmax}(\text{Score})
       $$
    4. **Weighted Sum**: The output is computed as a weighted sum of the Value vectors, using the softmax output as weights.
       $$
       \text{Output} = \text{Weights} \times V
       $$
- **Multi-Head Attention**: Multi-Head Attention allows the model to capture various types of relationships in the data by processing the input in parallel across different representation subspaces.
  - **Computation**:
       $$
    \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, ..., \text{head}_h)W^O
       $$
    Where each head is computed as:
       $$
    \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
       $$
- **Scaled Dot-Product Attention**: Scaled Dot-Product Attention is a mechanism used in the self-attention process, particularly within the Transformer model architecture. It incorporates a scaling factor to normalize the dot products, preventing overly large values that could lead to gradient vanishing during the Softmax calculation.


$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V $$


# Transformer

The Transformer model, introduced by Vaswani et al. in the paper "Attention is All You Need", revolutionizes sequence modeling by relying entirely on attention mechanisms, omitting recurrent or convolutional layers. This design allows for significantly increased training speeds and improved handling of long-range dependencies.

### 1. **Attention Mechanisms**
Transformers utilize scaled dot-product attention, which computes outputs by applying attention to all positions in the input sequence simultaneously. This parallel processing capability significantly enhances the efficiency and scalability of the model.

### 2. **Multi-Head Attention**
Instead of one single attention function, Transformers use multi-head attention which allows the model to jointly attend to information from different representation subspaces at different positions, capturing a richer variety of information.

### 3. **Positional Encoding**
Since Transformers do not inherently process sequential data, positional encodings are added to input embeddings to give the model information about the order of the sequence. These encodings use sine and cosine functions to encode positional information.

### 4. **Layer Normalization**
Layer normalization is applied before each sub-layer (attention and feed-forward networks) and residuals are added after each sub-layer, which helps stabilize the training process.

### 5. **Feed-Forward Networks**
In each layer, after attention processing, the data goes through a feed-forward neural network, which is applied independently to each position and identical across different positions.

## Advantages
- **Parallelization**: Unlike RNNs, which require sequential data processing, Transformers process all input data simultaneously, making them suitable for modern parallel processing hardware.
- **Handling Long-Range Dependencies**: Attention mechanisms allow Transformers to handle long-range dependencies in data more effectively than prior sequence models.

## Applications
Transformers have been highly successful in various NLP tasks like machine translation, text summarization, and question answering, setting new standards for model performance and versatility.



