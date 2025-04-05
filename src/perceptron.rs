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
}

impl Perceptron {
    pub fn new(datapoints: Vec<Datapoint>, weights: (f64, f64, f64)) -> Self {
        Perceptron {
            datapoints,
            weights: Vec3(weights.0, weights.1, weights.2),
        }
    }

    pub fn has_converged(&self) -> bool {
        false
    }

    pub fn update(&mut self) {
        let nn = NeuralNet::new(&self.datapoints, &self.weights);
        let learning_rate = 0.005;
        self.weights += learning_rate * -1.0 * nn.into_grad();
    }

    /// Return the weights.
    pub fn weights(&self) -> (f64, f64, f64) {
        (self.weights.0, self.weights.1, self.weights.2)
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

struct NeuralNet {
    loss: Value,
    w1: Value,
    w2: Value,
    w3: Value,
}

impl NeuralNet {
    fn get_label(x: f64, y: f64, w: &Vec3) -> i32 {
        let sum = w.0 + x * w.1 + y * w.2;
        let sigmoid = sum.exp() / (1.0 + sum.exp());
        if sigmoid <= 0.5 { 0 } else { 1 }
    }

    fn new(points: &Vec<Datapoint>, w: &Vec3) -> Self {
        let w1 = Value::from(w.0);
        let w2 = Value::from(w.1);
        let w3 = Value::from(w.2);
        let mut loss = Value::from(0.0);
        for point in points {
            let x1 = Value::from(1.0);
            let x2 = Value::from(point.pos.0 as f64);
            let x3 = Value::from(point.pos.1 as f64);
            let sum = x1 * w1.clone() + x2 * w2.clone() + x3 * w3.clone();
            let sigmoid = Value::from(1.0) / (Value::from(1.0) + (sum * (-1.0).into()).exp());
            let y = Value::from(point.label as f64);
            let single_loss = (y - sigmoid.clone()).pow(2.0);
            println!(
                "{point:?}, sigmoid={:0.2} loss={:0.2}",
                sigmoid.as_f64(),
                single_loss.as_f64()
            );
            loss = loss + single_loss;
        }
        loss = loss / Value::from(points.len() as f64);

        NeuralNet { loss, w1, w2, w3 }
    }

    fn into_grad(mut self) -> Vec3 {
        self.loss.backward();
        if self.w1.grad().is_finite() && self.w2.grad().is_finite() && self.w3.grad().is_finite() {
            let grad = Vec3(self.w1.grad(), self.w2.grad(), self.w3.grad());
            println!("LOSS {} GRAD {:0.2?}", self.loss.as_f64(), grad);
            grad
        } else {
            println!("LOSS INFINITE GRAD {}", self.loss.as_f64());
            Vec3::default()
        }
    }
}
