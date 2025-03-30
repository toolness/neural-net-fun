use std::ops::Add;

#[derive(Debug)]
pub enum Value {
    Float(f64),
    Sum(Box<Value>, Box<Value>),
}

impl From<f64> for Value {
    fn from(val: f64) -> Self {
        Value::Float(val)
    }
}

impl Value {
    fn compute(&self) -> f64 {
        match self {
            Value::Float(value) => *value,
            Value::Sum(a, b) => a.compute() + b.compute(),
        }
    }
}

impl Add<Value> for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        Value::Sum(Box::new(self), Box::new(rhs))
    }
}

fn main() {
    let sum = Value::Float(5.0) + 3.0.into();
    println!("hi {sum:?} = {}", sum.compute());
}
