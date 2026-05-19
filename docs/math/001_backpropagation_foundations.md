# Neural Network Mathematics: Backpropagation Foundations

## Table of Contents

1. [Neural Networks as Function Approximators](#1-neural-networks-as-function-approximators)
2. [The Perceptron](#2-the-perceptron)
3. [Activation Functions](#3-activation-functions)
4. [Multi-Layer Perceptron (MLP)](#4-multi-layer-perceptron-mlp)
5. [Loss Functions](#5-loss-functions)
6. [Optimization via Gradient Descent](#6-optimization-via-gradient-descent)
7. [The Chain Rule](#7-the-chain-rule)
8. [Backpropagation: Single Neuron](#8-backpropagation-single-neuron)
9. [Matrix Calculus for Layers](#9-matrix-calculus-for-layers)
10. [Full Forward and Backward Pass](#10-full-forward-and-backward-pass)
11. [Batch Processing](#11-batch-processing)
12. [Summary of Key Equations](#12-summary-of-key-equations)

---

## 1. Neural Networks as Function Approximators

A neural network is a parametric function that learns a mapping from inputs to outputs:

$$f_\theta(x) \rightarrow \hat{y}$$

Where:
- $x$ = input
- $\hat{y}$ = predicted output
- $\theta$ = learned parameters (weights and biases)

Training adjusts $\theta$ so that $\hat{y}$ approximates the true output $y$.

---

## 2. The Perceptron

A single perceptron computes a weighted sum of its inputs, adds a bias, then applies an activation function.

**Inputs and weights:**

$$x_1, x_2, \ldots, x_n \qquad w_1, w_2, \ldots, w_n$$

**Pre-activation (weighted sum):**

$$z = \sum_{i=1}^{n} w_i x_i + b = w^T x + b$$

**Activation:**

$$a = \phi(z)$$

Where $z$ is the raw output and $a$ is the activated output.

**Decision boundary** (2D case): The set of points where $z = 0$ defines a separating hyperplane:

$$w_1 x_1 + w_2 x_2 + b = 0$$

Generalizes to a plane in 3D and a hyperplane in higher dimensions.

---

## 3. Activation Functions

### Step Function

$$\phi(z) = \begin{cases} 1 & z \ge 0 \\ 0 & z < 0 \end{cases}$$

Not differentiable — cannot be used with gradient descent.

---

### Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Output range: $(0, 1)$. Used for binary classification output layers.

**Derivative:**

$$\frac{d\sigma}{dx} = \sigma(x)\bigl(1 - \sigma(x)\bigr)$$

*Derivation:* Let $u = 1 + e^{-x}$, then $\sigma = u^{-1}$ and $\frac{d\sigma}{dx} = -u^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1+e^{-x})^2} = \sigma(x)(1-\sigma(x))$.

---

### Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

Output range: $(-1, 1)$. Zero-centered, which can help gradient flow compared to sigmoid.

**Derivative:**

$$\frac{d\tanh}{dx} = 1 - \tanh^2(x)$$

*Derivation:* Using the quotient rule and the identity $\cosh^2 x - \sinh^2 x = 1$, we get $\text{sech}^2(x) = 1 - \tanh^2(x)$.

---

### ReLU

$$\text{ReLU}(x) = \max(0, x)$$

**Derivative:**

$$\frac{d\,\text{ReLU}}{dx} = \begin{cases} 1 & x > 0 \\ 0 & x \le 0 \end{cases}$$

Computationally cheap and avoids the vanishing gradient problem in deep networks.

---

### Softmax

$$\text{Softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

Produces a probability distribution. Used as the final activation for multi-class classification.

**Numerically stable form** (subtract max before exponentiation):

$$\text{Softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

---

## 4. Multi-Layer Perceptron (MLP)

An MLP stacks perceptron layers. For a 2-layer network:

**Layer 1 (hidden):**

$$z^{(1)} = W^{(1)} x + b^{(1)}$$
$$a^{(1)} = \phi\!\left(z^{(1)}\right)$$

**Layer 2 (output):**

$$z^{(2)} = W^{(2)} a^{(1)} + b^{(2)}$$
$$\hat{y} = \phi\!\left(z^{(2)}\right)$$

**Matrix dimensions** follow the rule $(m \times n)(n \times p) = (m \times p)$:

$$W_{4 \times 3} \, x_{3 \times 1} = z_{4 \times 1}$$

---

## 5. Loss Functions

The loss measures the gap between prediction and target.

**Mean Squared Error (batch):**

$$L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Single-sample version:**

$$L = \frac{1}{2}(a - y)^2$$

The $\frac{1}{2}$ factor cancels the exponent when differentiating, simplifying gradient expressions.

---

## 6. Optimization via Gradient Descent

**Objective:** find weights that minimize the loss:

$$W^* = \arg\min_W L(W)$$

**Weight update rule:**

$$w \leftarrow w - \eta \frac{\partial L}{\partial w}$$

Where $\eta$ is the **learning rate** — controls step size along the gradient.

---

## 7. The Chain Rule

Backpropagation is the repeated application of the chain rule through a computation graph.

For $y = f(g(x))$:

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

For a chain of $n$ functions $y = f_n(f_{n-1}(\cdots f_1(x) \cdots))$:

$$\frac{dy}{dx} = \frac{dy}{df_n} \cdot \frac{df_n}{df_{n-1}} \cdots \frac{df_2}{df_1} \cdot \frac{df_1}{dx}$$

**Computation graph view:**

```
Forward:   Input → Linear → Activation → Loss
Backward:  ∂L    ← ∂Act   ← ∂Linear   ← ∂Input
```

---

## 8. Backpropagation: Single Neuron

Given a single neuron with sigmoid activation and MSE loss:

$$z = wx + b \qquad a = \sigma(z) \qquad L = \tfrac{1}{2}(a - y)^2$$

Dependency chain: $w \rightarrow z \rightarrow a \rightarrow L$

**Step 1 — derivative of loss w.r.t. activation:**

$$\frac{\partial L}{\partial a} = a - y$$

**Step 2 — derivative of sigmoid:**

$$\frac{\partial a}{\partial z} = a(1 - a)$$

**Step 3 — derivative of weighted sum:**

$$\frac{\partial z}{\partial w} = x \qquad \frac{\partial z}{\partial b} = 1$$

**Weight gradient (chain rule):**

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w} = (a - y)\,a(1-a)\,x$$

**Bias gradient:**

$$\frac{\partial L}{\partial b} = (a - y)\,a(1-a)$$

**Error signal** $\delta$ (shorthand for the local gradient at $z$):

$$\delta = \frac{\partial L}{\partial z} = (a - y)\,a(1-a)$$

Then: $\dfrac{\partial L}{\partial w} = \delta x$ and $\dfrac{\partial L}{\partial b} = \delta$.

---

## 9. Matrix Calculus for Layers

For a layer with weight matrix $W$, input $X$, and bias $B$:

$$Z = WX + B \qquad A = \phi(Z)$$

### Output Layer Gradients

**Error signal at output layer $L$:**

$$dZ^{[L]} = (A^{[L]} - Y) \odot \phi'\!\left(Z^{[L]}\right)$$

Where $\odot$ denotes elementwise (Hadamard) multiplication.

**Weight gradient:**

$$dW^{[l]} = dZ^{[l]}\,(A^{[l-1]})^T$$

Interpretation: gradient = (error signal) × (previous layer activations).

**Bias gradient:**

$$dB^{[l]} = dZ^{[l]}$$

For batches, $dB$ is typically summed across samples.

**Input gradient (propagated upstream):**

$$dX^{[l]} = (W^{[l]})^T dZ^{[l]}$$

### Hidden Layer Backpropagation

To propagate the error signal back through a hidden layer:

$$dZ^{[l]} = (W^{[l+1]})^T dZ^{[l+1]} \odot \phi'\!\left(Z^{[l]}\right)$$

1. Multiply the downstream error by the transposed weight matrix (routes error back).
2. Apply elementwise multiplication by the local activation derivative (scales by sensitivity).

---

## 10. Full Forward and Backward Pass

**Forward pass** through layer $l$:

$$Z^{[l]} = W^{[l]} A^{[l-1]} + B^{[l]}$$
$$A^{[l]} = \phi\!\left(Z^{[l]}\right)$$

Where $A^{[0]} = X$ (the input).

**Backward pass** (reverse order):

| Quantity | Formula |
| -------- | ------- |
| Output error | $dZ^{[L]} = (A^{[L]} - Y) \odot \phi'(Z^{[L]})$ |
| Hidden error | $dZ^{[l]} = (W^{[l+1]})^T dZ^{[l+1]} \odot \phi'(Z^{[l]})$ |
| Weight gradient | $dW^{[l]} = dZ^{[l]} (A^{[l-1]})^T$ |
| Bias gradient | $dB^{[l]} = dZ^{[l]}$ |

**Weight update:**

$$W^{[l]} \leftarrow W^{[l]} - \eta\, dW^{[l]}$$

---

## 11. Batch Processing

Processing $m$ samples simultaneously:

$$X \in \mathbb{R}^{n \times m} \qquad \text{($n$ features, $m$ samples)}$$

Forward propagation is identical in form — matrix multiplication naturally handles the batch dimension.

**Batch-averaged weight gradient:**

$$dW = \frac{1}{m} dZ \, A^T$$

Dividing by $m$ makes the gradient magnitude independent of batch size, stabilizing learning rates across different batch sizes.

---

## 12. Summary of Key Equations

### Forward Propagation

$$Z^{[l]} = W^{[l]} A^{[l-1]} + B^{[l]}$$
$$A^{[l]} = \phi\!\left(Z^{[l]}\right)$$

### Backpropagation

$$dZ^{[L]} = (A^{[L]} - Y) \odot \phi'(Z^{[L]})$$
$$dZ^{[l]} = (W^{[l+1]})^T dZ^{[l+1]} \odot \phi'(Z^{[l]})$$
$$dW^{[l]} = dZ^{[l]}(A^{[l-1]})^T$$
$$dB^{[l]} = dZ^{[l]}$$

### Parameter Update

$$W \leftarrow W - \eta\, dW$$

### Activation Derivatives

| Activation | $\phi(x)$ | $\phi'(x)$ |
| ---------- | --------- | ---------- |
| Sigmoid | $\frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ |
| Tanh | $\tanh(x)$ | $1 - \tanh^2(x)$ |
| ReLU | $\max(0, x)$ | $\mathbf{1}[x > 0]$ |

### Everything Reduces To

$$\text{Linear Algebra} + \text{Calculus (Chain Rule)} + \text{Optimization (Gradient Descent)}$$
