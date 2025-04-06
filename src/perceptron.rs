use std::{
    fmt::Display,
    ops::{Add, Div, Mul},
};

use crate::{plot::Plot, value::Value};

use macroquad::{prelude::*, rand::rand};

#[derive(Clone, Copy, Debug)]
pub struct Datapoint {
    pub pos: (i32, i32),
    pub label: i32,
}

impl Datapoint {
    pub fn new(pos: (i32, i32), label: i32) -> Self {
        Datapoint { pos, label }
    }
}

#[derive(Clone, Debug)]
pub struct Weights(MultiLayerPerceptron<Value>);

fn rand_f64() -> f64 {
    (rand() as f64 / u32::MAX as f64) * 2.0 - 1.0
}

impl Weights {
    pub fn random() -> Self {
        Self(MultiLayerPerceptron::new(
            2,
            ActivationType::Sigmoid,
            vec![4, 1],
        ))
    }

    fn calculate_loss(&self, points: &Vec<Datapoint>, calc_grad: bool) -> Value {
        let mut loss = Value::from(0.0);
        for point in points {
            let inputs = vec![
                Value::from(point.pos.0 as f64),
                Value::from(point.pos.1 as f64),
            ];
            let output = self.0.output(&inputs).pop().unwrap();
            let y = Value::from(point.label as f64);
            let single_loss = (y - output.clone()).pow(2.0);
            // println!(
            //     "{point:?}, sigmoid={:0.2} loss={:0.2}",
            //     sigmoid.as_f64(),
            //     single_loss.as_f64()
            // );
            loss = loss + single_loss;
        }
        loss = loss / Value::from(points.len() as f64);

        if calc_grad {
            for param in self.0.params().iter_mut() {
                param.zero_grad();
            }
            loss.backward();
        }

        loss
    }

    fn learn(&self, learning_rate: f64) {
        for mut value in self.0.params() {
            let mut new_f64 = value.as_f64() + -1.0 * learning_rate * value.grad();
            // Reset any infinite weights.
            if !new_f64.is_finite() {
                new_f64 = rand_f64();
            }
            value.set(new_f64);
        }
    }
}

impl Display for Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let floats: Vec<String> = self
            .0
            .params()
            .iter()
            .map(|v| format!("{:.2}", v.as_f64()))
            .collect();
        write!(f, "({})", floats.join(", "))
    }
}

pub struct Perceptron {
    datapoints: Vec<Datapoint>,
    weights: Weights,
    loss: f64,
}

impl Perceptron {
    pub fn new(datapoints: Vec<Datapoint>, weights: Weights) -> Self {
        let loss = weights.calculate_loss(&datapoints, false).as_f64();
        Perceptron {
            datapoints,
            weights,
            loss,
        }
    }

    pub fn has_converged(&self) -> bool {
        self.loss < 0.0001
    }

    pub fn update(&mut self, learning_rate: f64) {
        self.loss = self.weights.calculate_loss(&self.datapoints, true).as_f64();
        self.weights.learn(learning_rate);
        //println!("new weights: {}", self.weights);
    }

    /// Return the weights.
    pub fn weights(&self) -> Weights {
        self.weights.clone()
    }

    pub fn loss(&self) -> f64 {
        self.loss
    }

    pub fn draw(&self, plot: &Plot) {
        let mlp = self.weights.0.read_only();

        for y in -50..50 {
            for x in -50..50 {
                let inputs = vec![x as f64, y as f64];
                let output = *mlp.output(&inputs).first().unwrap();
                let label = if output <= 0.5 { 0 } else { 1 };

                plot.draw_point(
                    x as f32,
                    y as f32,
                    if label <= 0 { DARKPURPLE } else { DARKGREEN },
                );
            }
        }

        // Draw datapoints.
        for (_index, point) in self.datapoints.iter().enumerate() {
            plot.draw_circle(
                point.pos.0 as f32,
                point.pos.1 as f32,
                0.5,
                if point.label <= 0 { PURPLE } else { GREEN },
            );
        }
    }
}

trait NeuronValue:
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

#[derive(Clone, Debug)]
struct MultiLayerPerceptron<V: NeuronValue> {
    layers: Vec<Layer<V>>,
}

impl<V: NeuronValue> MultiLayerPerceptron<V> {
    fn new(num_inputs: usize, activation: ActivationType, num_layer_outputs: Vec<usize>) -> Self {
        let mut layers = vec![];
        let mut next_num_inputs = num_inputs;
        for num_outputs in num_layer_outputs {
            layers.push(Layer::new(next_num_inputs, activation, num_outputs));
            next_num_inputs = num_outputs;
        }
        Self { layers }
    }

    fn read_only(&self) -> MultiLayerPerceptron<f64> {
        MultiLayerPerceptron {
            layers: self.layers.iter().map(|layer| layer.read_only()).collect(),
        }
    }

    fn output(&self, inputs: &Vec<V>) -> Vec<V> {
        let mut next_inputs = inputs.clone();
        for layer in &self.layers {
            let layer_outputs = layer.output(&next_inputs);
            next_inputs = layer_outputs;
        }
        next_inputs
    }

    fn params(&self) -> Vec<V> {
        self.layers
            .iter()
            .flat_map(|layer| layer.params())
            .collect()
    }
}
