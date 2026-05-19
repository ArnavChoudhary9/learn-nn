# Neural Network Mathematics — Foundations of Backpropagation

# 1. Neural Networks as Functions

A neural network is fundamentally a function approximator.

\[
f(x) \rightarrow y
\]

Where:

- \(x\) = input
- \(y\) = desired output
- \(f\) = learned mapping

---

# 2. Perceptron Mathematics

A perceptron computes a weighted sum of inputs.

Inputs:

\[
x_1, x_2, x_3, ..., x_n
\]

Weights:

\[
w_1, w_2, w_3, ..., w_n
\]

Bias:

\[
b
\]

Weighted sum:

\[
z = \sum_{i=1}^{n} w_i x_i + b
\]

Activation:

\[
a = \phi(z)
\]

Where:
- \(z\) = raw output
- \(a\) = activated output

---

# 3. Decision Boundary

For 2-dimensional input:

\[
w_1x_1 + w_2x_2 + b = 0
\]

This defines a line separating classes.

Higher dimensions:
- plane
- hyperplane

---

# 4. Activation Functions

## Step Function

\[
\phi(z) =
\begin{cases}
1 & z \ge 0 \\
0 & z < 0
\end{cases}
\]

---

## Sigmoid

\[
\sigma(x)=\frac{1}{1+e^{-x}}
\]

Derivative:

\[
\frac{d\sigma}{dx}=\sigma(x)(1-\sigma(x))
\]

---

## Tanh

\[
\tanh(x)
\]

---

## ReLU

\[
\text{ReLU}(x)=\max(0,x)
\]

---

# 5. Multi-Layer Perceptron (MLP)

Hidden layer computation:

\[
z^{(1)} = W^{(1)}x + b^{(1)}
\]

Activation:

\[
a^{(1)} = \phi(z^{(1)})
\]

Output layer:

\[
z^{(2)} = W^{(2)}a^{(1)} + b^{(2)}
\]

Final prediction:

\[
\hat{y} = \phi(z^{(2)})
\]

---

# 6. Matrix Representation

Forward propagation:

\[
z = Wx + b
\]

Where:
- \(W\) = weight matrix
- \(x\) = input vector
- \(b\) = bias vector

---

# 7. Loss Function

Mean Squared Error:

\[
L = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

Single-sample version:

\[
L = \frac12(a-y)^2
\]

---

# 8. Optimization Objective

Goal:

\[
W^* = \arg\min_W L(W)
\]

Meaning:
find weights minimizing loss.

---

# 9. Gradient Descent

Weight update rule:

\[
w := w - \eta \frac{\partial L}{\partial w}
\]

Where:
- \(\eta\) = learning rate

---

# 10. Chain Rule

If:

\[
y=f(g(x))
\]

Then:

\[
\frac{dy}{dx}
=
\frac{dy}{dg}
\cdot
\frac{dg}{dx}
\]

This powers backpropagation.

---

# 11. Single Neuron Backpropagation

Neuron:

\[
z = wx+b
\]

Activation:

\[
a=\sigma(z)
\]

Loss:

\[
L=\frac12(a-y)^2
\]

Dependency chain:

\[
w \rightarrow z \rightarrow a \rightarrow L
\]

---

# 12. Derivative of Loss wrt Activation

\[
\frac{\partial L}{\partial a}=a-y
\]

---

# 13. Derivative of Sigmoid

\[
\frac{\partial a}{\partial z}=a(1-a)
\]

---

# 14. Derivative of Weighted Sum

\[
\frac{\partial z}{\partial w}=x
\]

Bias derivative:

\[
\frac{\partial z}{\partial b}=1
\]

---

# 15. Final Weight Gradient

Using chain rule:

\[
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial a}
\cdot
\frac{\partial a}{\partial z}
\cdot
\frac{\partial z}{\partial w}
\]

Substitute values:

\[
\frac{\partial L}{\partial w}
=
(a-y)a(1-a)x
\]

---

# 16. Bias Gradient

\[
\frac{\partial L}{\partial b}
=
(a-y)a(1-a)
\]

---

# 17. Error Signal

Define:

\[
\delta
=
\frac{\partial L}{\partial z}
\]

For sigmoid output neuron:

\[
\delta=(a-y)a(1-a)
\]

Then:

\[
\frac{\partial L}{\partial w}=\delta x
\]

---

# 18. Matrix Calculus for Neural Networks

Layer equation:

\[
Z = WX + B
\]

Activation:

\[
A=\phi(Z)
\]

---

# 19. Shape Rules

Matrix multiplication rule:

\[
(m\times n)(n\times p)=(m\times p)
\]

Example:

\[
W_{4\times3}x_{3\times1}=z_{4\times1}
\]

---

# 20. Full Forward Propagation

Layer \(l\):

\[
Z^{[l]} = W^{[l]}A^{[l-1]} + B^{[l]}
\]

Activation:

\[
A^{[l]}=\phi(Z^{[l]})
\]

---

# 21. Output Layer Backpropagation

For output layer:

\[
dZ^{[L]}
=
(A^{[L]}-Y)
\odot
\sigma'(Z^{[L]})
\]

Where:
- \(\odot\) = elementwise multiplication

---

# 22. Weight Gradient (Matrix Form)

\[
dW^{[l]}
=
dZ^{[l]}(A^{[l-1]})^T
\]

Interpretation:

\[
\text{gradient}=
\text{error signal}
\times
\text{previous activations}
\]

---

# 23. Bias Gradient (Matrix Form)

\[
dB^{[l]} = dZ^{[l]}
\]

For batches:
typically summed across samples.

---

# 24. Hidden Layer Backpropagation

Core hidden layer equation:

\[
dZ^{[l]}
=
(W^{[l+1]})^T
dZ^{[l+1]}
\odot
\phi'(Z^{[l]})
\]

Meaning:

1. propagate downstream error backward
2. multiply by local activation sensitivity

---

# 25. Batch Processing

Input matrix:

\[
X \in \mathbb{R}^{n\times m}
\]

Where:
- \(n\) = features
- \(m\) = batch size

Forward propagation:

\[
Z = WX + B
\]

---

# 26. Batch Gradient Formula

Average batch gradient:

\[
dW=\frac1m dZA^T
\]

---

# 27. Important Neural Network Equations

## Forward Propagation

\[
Z^{[l]} = W^{[l]}A^{[l-1]} + B^{[l]}
\]

\[
A^{[l]}=\phi(Z^{[l]})
\]

---

## Hidden Layer Backpropagation

\[
dZ^{[l]}
=
(W^{[l+1]})^T
dZ^{[l+1]}
\odot
\phi'(Z^{[l]})
\]

---

## Weight Gradient

\[
dW^{[l]}
=
dZ^{[l]}(A^{[l-1]})^T
\]

---

## Weight Update

\[
W := W - \eta dW
\]

---

# 28. Computational Graph Interpretation

Forward pass:

\[
\text{Input}
\rightarrow
\text{Linear}
\rightarrow
\text{Activation}
\rightarrow
\text{Loss}
\]

Backward pass:

\[
\text{Loss Gradient}
\rightarrow
\text{Activation Gradient}
\rightarrow
\text{Linear Gradient}
\]

Backpropagation is:

\[
\text{Repeated application of chain rule through a computation graph}
\]

---

# 29. Key Concepts Summary

Neural networks fundamentally rely on:

- Linear algebra
- Matrix multiplication
- Partial derivatives
- Chain rule
- Optimization
- Gradient descent
- Function composition

Everything reduces to:

\[
\text{Linear Algebra}
+
\text{Calculus}
+
\text{Optimization}
\]
