use std::ops::{Add, Div, Mul};

use macroquad::rand::rand;

use crate::value::Value;

/// Returns a random floating-point number between -1 and 1,
/// I *think* it's inclusive but I'm not 100% sure (I wish
/// macroquad's documentation was more specific about this
/// but it doesn't seem to be).
pub fn rand_f64() -> f64 {
    (rand() as f64 / u32::MAX as f64) * 2.0 - 1.0
}

/// A trait that represents the underlying value used by a
/// neuron. If backprop is a concern, the implementation in
/// `Value` can be used, but otherwise the `f64`
/// implementation is much more efficient and [Send]-able.
pub trait NeuronValue:
    Clone
    + std::fmt::Debug
    + From<f64>
    + Mul<Self, Output = Self>
    + Add<Self, Output = Self>
    + Div<Self, Output = Self>
{
    fn exp(&self) -> Self;

    fn as_f64(&self) -> f64;
}

impl NeuronValue for Value {
    fn exp(&self) -> Value {
        self.exp()
    }

    fn as_f64(&self) -> f64 {
        self.as_f64()
    }
}

impl NeuronValue for f64 {
    fn exp(&self) -> f64 {
        f64::exp(*self)
    }

    fn as_f64(&self) -> f64 {
        *self
    }
}

/// Represents an activation function for neurons.
#[derive(Clone, Copy, Debug)]
pub enum ActivationType {
    Sigmoid,
}

impl ActivationType {
    fn activate<V: NeuronValue>(&self, value: V) -> V {
        match self {
            ActivationType::Sigmoid => {
                V::from(1.0) / (V::from(1.0) + (value * (-1.0).into()).exp())
            }
        }
    }
}

/// A single neuron in a neural net. It can have any number of
/// inputs and always produces a single output value.
#[derive(Clone, Debug)]
struct Neuron<V: NeuronValue> {
    weights: Vec<V>,
    bias: V,
    activation: ActivationType,
}

impl<V: NeuronValue> Neuron<V> {
    fn new(num_inputs: usize, activation: ActivationType) -> Self {
        Neuron {
            weights: (0..num_inputs).map(|_| rand_f64().into()).collect(),
            bias: rand_f64().into(),
            activation,
        }
    }

    fn read_only(&self) -> Neuron<f64> {
        Neuron {
            weights: self.weights.iter().map(|weight| weight.as_f64()).collect(),
            bias: self.bias.as_f64(),
            activation: self.activation,
        }
    }

    fn output(&self, inputs: &Vec<V>) -> V {
        assert_eq!(self.weights.len(), inputs.len());
        let mut sum = self.bias.clone();
        for (weight, input) in self.weights.iter().zip(inputs) {
            sum = sum + weight.clone() * input.clone();
        }
        self.activation.activate(sum)
    }

    fn params(&self) -> Vec<V> {
        let mut params = self.weights.clone();
        params.push(self.bias.clone());
        params
    }
}

/// A layer in a neural net.
#[derive(Clone, Debug)]
struct Layer<V: NeuronValue> {
    neurons: Vec<Neuron<V>>,
}

impl<V: NeuronValue> Layer<V> {
    fn new(num_inputs: usize, activation: ActivationType, num_outputs: usize) -> Self {
        Layer {
            neurons: (0..num_outputs)
                .map(|_| Neuron::new(num_inputs, activation))
                .collect(),
        }
    }

    fn read_only(&self) -> Layer<f64> {
        Layer {
            neurons: self
                .neurons
                .iter()
                .map(|neuron| neuron.read_only())
                .collect(),
        }
    }

    fn output(&self, inputs: &Vec<V>) -> Vec<V> {
        self.neurons
            .iter()
            .map(|neuron| neuron.output(inputs))
            .collect()
    }

    fn params(&self) -> Vec<V> {
        self.neurons
            .iter()
            .flat_map(|neuron| neuron.params())
            .collect()
    }
}

/// A neural net with multiple layers, some of which may
/// be hidden.
#[derive(Clone, Debug)]
pub struct MultiLayerPerceptron<V: NeuronValue> {
    layers: Vec<Layer<V>>,
}

impl<V: NeuronValue> MultiLayerPerceptron<V> {
    pub fn new(
        num_inputs: usize,
        activation: ActivationType,
        num_layer_outputs: Vec<usize>,
    ) -> Self {
        let mut layers = vec![];
        let mut next_num_inputs = num_inputs;
        for num_outputs in num_layer_outputs {
            layers.push(Layer::new(next_num_inputs, activation, num_outputs));
            next_num_inputs = num_outputs;
        }
        Self { layers }
    }

    pub fn read_only(&self) -> MultiLayerPerceptron<f64> {
        MultiLayerPerceptron {
            layers: self.layers.iter().map(|layer| layer.read_only()).collect(),
        }
    }

    pub fn output(&self, inputs: &Vec<V>) -> Vec<V> {
        let mut next_inputs = inputs.clone();
        for layer in &self.layers {
            let layer_outputs = layer.output(&next_inputs);
            next_inputs = layer_outputs;
        }
        next_inputs
    }

    pub fn params(&self) -> Vec<V> {
        self.layers
            .iter()
            .flat_map(|layer| layer.params())
            .collect()
    }
}
