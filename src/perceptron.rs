use crate::{plot::Plot, value::Value};

use super::vec3::Vec3;
use macroquad::prelude::*;

#[derive(Clone, Copy)]
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
    /// The current index in `datapoints` that we're looking at next.
    curr_index: usize,
    /// The index in `datapoints` that caused the most recent weight update.
    last_update_index: Option<usize>,
    /// How many weight updates have been made in this generation of
    /// updates. A generation is an iteration through all the datapoints.
    updates_this_generation: usize,
    /// Whether or not the Perceptron has gone through an entire
    /// generation without needing to update its weights.
    has_converged: bool,
}

impl Perceptron {
    pub fn new(datapoints: Vec<Datapoint>, weights: (f64, f64, f64)) -> Self {
        Perceptron {
            datapoints,
            weights: Vec3(weights.0, weights.1, weights.2),
            curr_index: 0,
            updates_this_generation: 0,
            last_update_index: None,
            has_converged: false,
        }
    }

    /// Returns whether all future calls to `update` will do nothing.
    pub fn has_converged(&self) -> bool {
        self.has_converged
    }

    /// Try to make a single weight update to the Perceptron based on the
    /// next datapoint that the Perceptron doesn't classify correctly.
    ///
    /// Weights won't be updated if the Perceptron's solution has converged.
    pub fn update(&mut self) {
        loop {
            match self.datapoints.get(self.curr_index) {
                Some(point) => {
                    let nn = NeuralNet::new(point, &self.weights);
                    let learning_rate = 0.05;
                    let grad = nn.into_grad();
                    if true {
                        self.weights += learning_rate * -1.0 * grad;
                        self.updates_this_generation += 1;
                        self.last_update_index = Some(self.curr_index);
                        self.curr_index += 1;
                        return;
                    }
                    self.curr_index += 1;
                }
                None => {
                    if self.updates_this_generation == 0 {
                        self.has_converged = true;
                    } else {
                        self.updates_this_generation = 0;
                        self.curr_index = 0;
                        self.update();
                    }
                    return;
                }
            }
        }
    }

    fn get_point_color(&self, point: &Datapoint, index: usize) -> Color {
        let is_last_update = !self.has_converged && self.last_update_index == Some(index);
        if point.label <= 0 {
            if is_last_update { PURPLE } else { DARKPURPLE }
        } else {
            if is_last_update { GREEN } else { DARKGREEN }
        }
    }

    /// Return the weights.
    pub fn weights(&self) -> (f64, f64, f64) {
        (self.weights.0, self.weights.1, self.weights.2)
    }

    pub fn draw(&self, plot: &Plot) {
        for y in -50..50 {
            for x in -50..50 {
                let label = NeuralNet::forward(x as f64, y as f64, &self.weights);
                plot.draw_point(
                    x as f32,
                    y as f32,
                    if label <= 0 { DARKPURPLE } else { DARKGREEN },
                );
            }
        }

        // Draw datapoints.
        for (index, point) in self.datapoints.iter().enumerate() {
            plot.draw_circle(
                point.pos.0 as f32,
                point.pos.1 as f32,
                0.5,
                self.get_point_color(point, index),
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
    fn forward(x: f64, y: f64, w: &Vec3) -> i32 {
        let sum = w.0 + x * w.1 + y * w.2;
        let sigmoid = sum.exp() / (1.0 + sum.exp());
        if sigmoid <= 0.5 { -1 } else { 1 }
    }

    fn new(point: &Datapoint, w: &Vec3) -> Self {
        let x1 = Value::from(1.0);
        let x2 = Value::from(point.pos.0 as f64);
        let x3 = Value::from(point.pos.1 as f64);
        let w1 = Value::from(w.0);
        let w2 = Value::from(w.1);
        let w3 = Value::from(w.2);
        let sum = x1 * w1.clone() + x2 * w2.clone() + x3 * w3.clone();
        let sigmoid = sum.exp() / (Value::from(1.0) + sum.exp());
        let y = Value::from(if point.label == -1 { 0.0 } else { 1.0 });
        let loss = (y - sigmoid).pow(2.0);

        NeuralNet { loss, w1, w2, w3 }
    }

    fn into_grad(mut self) -> Vec3 {
        self.loss.backward();
        if self.w1.grad().is_finite() && self.w2.grad().is_finite() && self.w3.grad().is_finite() {
            println!("LOSS {}", self.loss.as_f64());
            Vec3(self.w1.grad(), self.w2.grad(), self.w3.grad())
        } else {
            println!("LOSS INFINITE GRAD {}", self.loss.as_f64());
            Vec3::default()
        }
    }
}
