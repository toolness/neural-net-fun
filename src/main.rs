use std::{cell::RefCell, ops::Add, rc::Rc};

#[derive(Debug, Clone)]
pub struct Param(Rc<RefCell<f64>>);

impl Param {
    fn set(&mut self, value: f64) {
        *self.0.borrow_mut() = value;
    }

    fn get(&self) -> f64 {
        *self.0.borrow()
    }
}

impl From<f64> for Param {
    fn from(val: f64) -> Self {
        Param(Rc::new(val.into()))
    }
}

#[derive(Debug)]
pub enum Value {
    Constant(f64),
    Param(Param),
    Sum(Box<Value>, Box<Value>),
}

impl From<f64> for Value {
    fn from(val: f64) -> Self {
        Value::Constant(val)
    }
}

impl From<Param> for Value {
    fn from(val: Param) -> Self {
        Value::Param(val)
    }
}

impl Value {
    fn compute(&self) -> f64 {
        match self {
            Value::Constant(value) => *value,
            Value::Param(value) => value.get(),
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
    let mut x = Param::from(1.0);
    let sum = Value::Param(x.clone()) + 3.0.into();
    println!("{sum:?} = {}", sum.compute());
    x.set(2.0);
    println!("{sum:?} = {}", sum.compute());
}
