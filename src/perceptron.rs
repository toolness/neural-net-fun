use std::fmt::Display;

use crate::{
    engine::{ActivationType, MultiLayerPerceptron, rand_f64},
    plot::Plot,
    value::Value,
};

use macroquad::prelude::*;
use rayon::prelude::*;

/// Arg, we need this to normalize the datapoints so they're roughly within (-1, 1),
/// as this will make the data much easier to fit.
const POINT_SCALE: f64 = 30.0;

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

impl Weights {
    pub fn new(mut hidden_layers: Vec<usize>) -> Self {
        hidden_layers.push(1);
        Self(MultiLayerPerceptron::new(
            2,
            ActivationType::Sigmoid,
            hidden_layers,
        ))
    }

    fn calculate_loss(&self, points: &Vec<Datapoint>, calc_grad: bool) -> Value {
        let mut loss = Value::from(0.0);
        for point in points {
            let inputs = vec![
                Value::from(point.pos.0 as f64 / POINT_SCALE),
                Value::from(point.pos.1 as f64 / POINT_SCALE),
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

    pub fn draw(&self, plot: &Plot, enable_shading: bool) {
        let mlp = self.weights.0.read_only();

        let all_points = (-50..50)
            .flat_map(|y| (-50..50).map(move |x| (x, y)))
            .collect::<Vec<_>>();

        let points = all_points
            .into_par_iter()
            .map(|(x, y)| {
                let inputs = vec![x as f64 / POINT_SCALE, y as f64 / POINT_SCALE];
                let output = *mlp.output(&inputs).first().unwrap();
                let color = if output <= 0.5 {
                    DARKPURPLE.with_alpha(if enable_shading {
                        1.0 - output as f32 * 2.0
                    } else {
                        1.0
                    })
                } else {
                    DARKGREEN.with_alpha(if enable_shading {
                        (output as f32 - 0.5) * 2.0
                    } else {
                        1.0
                    })
                };
                (x, y, color)
            })
            .collect::<Vec<_>>();

        for (x, y, color) in points {
            plot.draw_point(x as f32, y as f32, color);
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
