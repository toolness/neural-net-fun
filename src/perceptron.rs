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
pub struct Weights(Neuron);

fn rand_f64() -> f64 {
    (rand() as f64 / u32::MAX as f64) * 2.0 - 1.0
}

fn rand_value() -> Value {
    Value::from(rand_f64())
}

impl Weights {
    pub fn random() -> Self {
        Self(Neuron::new(2, sigmoid))
    }

    fn get_label(&self, x: f64, y: f64) -> i32 {
        let inputs = vec![Value::from(x), Value::from(y)];
        let output = self.0.output(&inputs).as_f64();
        if output <= 0.5 { 0 } else { 1 }
    }

    fn calculate_loss(&self, points: &Vec<Datapoint>, calc_grad: bool) -> Value {
        let mut loss = Value::from(0.0);
        for point in points {
            let inputs = vec![
                Value::from(point.pos.0 as f64),
                Value::from(point.pos.1 as f64),
            ];
            let output = self.0.output(&inputs);
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
        false
    }

    pub fn update(&mut self, learning_rate: f64) {
        self.loss = self.weights.calculate_loss(&self.datapoints, true).as_f64();
        self.weights.learn(learning_rate);
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
                let label = self.weights.get_label(x as f64, y as f64);
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

#[derive(Clone, Debug)]
struct Neuron {
    weights: Vec<Value>,
    bias: Value,
    activation: ActivationFn,
}

impl Neuron {
    fn new(num_inputs: usize, activation: ActivationFn) -> Self {
        Neuron {
            weights: (0..num_inputs).map(|_| rand_value()).collect(),
            bias: rand_value(),
            activation,
        }
    }

    fn output(&self, inputs: &Vec<Value>) -> Value {
        assert_eq!(self.weights.len(), inputs.len());
        let mut sum = self.bias.clone();
        for (weight, input) in self.weights.iter().zip(inputs) {
            sum = sum + weight.clone() * input.clone();
        }
        (self.activation)(sum)
    }

    fn params(&self) -> Vec<Value> {
        let mut params = self.weights.clone();
        params.push(self.bias.clone());
        params
    }
}
