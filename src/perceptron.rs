use std::fmt::Display;

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
pub struct Weights(Vec<Value>);

fn rand_f64() -> f64 {
    (rand() as f64 / u32::MAX as f64) * 2.0 - 1.0
}

fn rand_value() -> Value {
    Value::from(rand_f64())
}

impl Weights {
    pub fn random() -> Self {
        let weights = vec![rand_value(), rand_value(), rand_value()];
        Self(weights)
    }

    fn learn(&self, grad: Vec<f64>, learning_rate: f64) -> Self {
        assert_eq!(self.0.len(), grad.len());
        Weights(
            self.0
                .iter()
                .zip(grad)
                .map(|(value, grad)| {
                    let mut new_f64 = value.as_f64() + -1.0 * learning_rate * grad;
                    // Reset any infinite weights.
                    if !new_f64.is_finite() {
                        new_f64 = rand_f64();
                    }
                    Value::from(new_f64)
                })
                .collect(),
        )
    }
}

impl Display for Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let floats: Vec<String> = self
            .0
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
        let mut p = Perceptron {
            datapoints,
            weights,
            loss: 0.0,
        };
        p.create_nn_and_update_loss();
        p
    }

    pub fn has_converged(&self) -> bool {
        false
    }

    fn create_nn_and_update_loss(&mut self) -> NeuralNet {
        let nn = NeuralNet::new(&self.datapoints, &self.weights);
        self.loss = nn.loss.as_f64();
        nn
    }

    pub fn update(&mut self) {
        let learning_rate = 0.05;
        let nn = self.create_nn_and_update_loss();
        let grad = nn.into_grad();
        self.weights = self.weights.learn(grad, learning_rate);
        println!("new weights: {}", self.weights);
    }

    /// Return the weights.
    pub fn weights(&self) -> Weights {
        self.weights.clone()
    }

    pub fn loss(&self) -> f64 {
        self.loss
    }

    pub fn draw(&self, plot: &Plot) {
        for y in -50..50 {
            for x in -50..50 {
                let label = NeuralNet::get_label(x as f64, y as f64, &self.weights);
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

type ActivationFn = fn(value: Value) -> Value;

fn sigmoid(value: Value) -> Value {
    Value::from(1.0) / (Value::from(1.0) + (value * (-1.0).into()).exp())
}

fn neuron(weights: &Vec<Value>, inputs: &Vec<Value>, activation: ActivationFn) -> Value {
    // The weights will have one extra for the bias.
    assert_eq!(weights.len(), inputs.len() + 1);
    let mut sum = weights.last().unwrap().clone();
    for (weight, input) in weights.iter().zip(inputs) {
        sum = sum + weight.clone() * input.clone();
    }
    activation(sum)
}

struct NeuralNet {
    loss: Value,
    weights: Vec<Value>,
}

impl NeuralNet {
    fn get_label(x: f64, y: f64, w: &Weights) -> i32 {
        let inputs = vec![Value::from(x), Value::from(y)];
        let output = neuron(&w.0, &inputs, sigmoid).as_f64();
        if output <= 0.5 { 0 } else { 1 }
    }

    fn new(points: &Vec<Datapoint>, w: &Weights) -> Self {
        let weights = w.0.clone();
        let mut loss = Value::from(0.0);
        for point in points {
            let inputs = vec![
                Value::from(point.pos.0 as f64),
                Value::from(point.pos.1 as f64),
            ];
            let output = neuron(&weights, &inputs, sigmoid);
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

        NeuralNet { loss, weights }
    }

    fn into_grad(mut self) -> Vec<f64> {
        for weight in self.weights.iter_mut() {
            weight.zero_grad();
        }
        self.loss.backward();
        self.weights
            .into_iter()
            .map(|weight| weight.grad())
            .collect()
    }
}
