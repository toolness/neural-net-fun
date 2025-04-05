use crate::{plot::Plot, value::Value};

use super::vec3::Vec3;
use macroquad::prelude::*;

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

pub struct Perceptron {
    datapoints: Vec<Datapoint>,
    weights: Vec3,
    loss: f64,
}

impl Perceptron {
    pub fn new(datapoints: Vec<Datapoint>, weights: (f64, f64, f64)) -> Self {
        let mut p = Perceptron {
            datapoints,
            weights: Vec3(weights.0, weights.1, weights.2),
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
        let learning_rate = 0.005;
        let nn = self.create_nn_and_update_loss();
        self.weights += learning_rate * -1.0 * nn.into_grad();
    }

    /// Return the weights.
    pub fn weights(&self) -> (f64, f64, f64) {
        (self.weights.0, self.weights.1, self.weights.2)
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
    fn get_label(x: f64, y: f64, w: &Vec3) -> i32 {
        let sum = x * w.0 + y * w.1 + w.2;
        let sigmoid = sum.exp() / (1.0 + sum.exp());
        if sigmoid <= 0.5 { 0 } else { 1 }
    }

    fn new(points: &Vec<Datapoint>, w: &Vec3) -> Self {
        let weights = vec![Value::from(w.0), Value::from(w.1), Value::from(w.2)];
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

    fn into_grad(mut self) -> Vec3 {
        self.loss.backward();

        let w3 = self.weights.pop().unwrap();
        let w2 = self.weights.pop().unwrap();
        let w1 = self.weights.pop().unwrap();
        if w1.grad().is_finite() && w2.grad().is_finite() && w3.grad().is_finite() {
            let grad = Vec3(w1.grad(), w2.grad(), w3.grad());
            grad
        } else {
            // TODO: Consider resetting any weight that isn't finite.
            Vec3::default()
        }
    }
}
