use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::Display,
    ops::{Add, Mul},
    rc::Rc,
};

#[derive(Debug, Clone)]
pub struct Value(Rc<RefCell<InnerValue>>);

impl From<InnerValue> for Value {
    fn from(value: InnerValue) -> Self {
        Value(Rc::new(value.into()))
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        InnerValue::new(ValueType::Float(None), value).into()
    }
}

impl Value {
    pub fn as_f64(&self) -> f64 {
        self.0.borrow().value
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }
}

impl Value {
    pub fn new_param<T: AsRef<str>>(name: T, value: f64) -> Value {
        InnerValue::new(ValueType::Float(Some(name.as_ref().to_owned())), value).into()
    }

    fn children(&self) -> Vec<Value> {
        match &self.0.borrow()._type {
            ValueType::Float(_) => vec![],
            ValueType::Mul(a, b) => vec![a.clone(), b.clone()],
            ValueType::Sum(a, b) => vec![a.clone(), b.clone()],
        }
    }

    pub fn backward(&mut self) {
        self.0.borrow_mut().grad = 1.0;
        for mut value in BackpropIterator::new(self.clone()) {
            value.local_backward();
        }
    }

    fn local_backward(&mut self) {
        let value = &self.0.borrow();
        match &value._type {
            ValueType::Float(_) => {}
            ValueType::Sum(a, b) => {
                a.0.borrow_mut().grad += value.grad;
                b.0.borrow_mut().grad += value.grad;
            }
            ValueType::Mul(a, b) => {
                a.0.borrow_mut().grad += b.0.borrow().value * value.grad;
                b.0.borrow_mut().grad += a.0.borrow().value * value.grad;
            }
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.borrow())
    }
}

impl Add<Value> for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        InnerValue::new(
            ValueType::Sum(self.clone(), rhs.clone()),
            self.as_f64() + rhs.as_f64(),
        )
        .into()
    }
}

impl Mul<Value> for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Self::Output {
        InnerValue::new(
            ValueType::Mul(self.clone(), rhs.clone()),
            self.as_f64() * rhs.as_f64(),
        )
        .into()
    }
}

#[derive(Debug)]
enum ValueType {
    Float(Option<String>),
    Sum(Value, Value),
    Mul(Value, Value),
}

#[derive(Debug)]
struct InnerValue {
    _type: ValueType,
    value: f64,
    grad: f64,
}

impl InnerValue {
    fn new(_type: ValueType, value: f64) -> Self {
        InnerValue {
            _type,
            value,
            grad: 0.0,
        }
    }
}

impl Display for InnerValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self._type {
            ValueType::Float(Some(name)) => write!(f, "{name}"),
            ValueType::Float(None) => write!(f, "{}", self.value),
            ValueType::Sum(a, b) => write!(f, "({} + {})", a, b),
            ValueType::Mul(a, b) => write!(f, "({} * {})", a, b),
        }
    }
}

struct BackpropIterator {
    visited: HashSet<i64>,
    to_visit: Vec<Value>,
}

impl BackpropIterator {
    fn new(root: Value) -> Self {
        BackpropIterator {
            visited: HashSet::new(),
            to_visit: vec![root],
        }
    }
}

impl Iterator for BackpropIterator {
    type Item = Value;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(value) = self.to_visit.pop() {
            let ptr = Rc::as_ptr(&value.0) as i64;
            if self.visited.contains(&ptr) {
                continue;
            }
            self.to_visit.extend(value.children());
            return Some(value.clone());
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use crate::value::Value;

    #[test]
    fn test_karpathy_example() {
        // This example is taken from Karpathy's lecture at around 51:38:
        // https://www.youtube.com/watch?v=VMj-3S1tku0
        let a = Value::new_param("a", 2.0);
        let b = Value::new_param("b", -3.0);
        let c = Value::new_param("c", 10.0);
        let e = a.clone() * b.clone();
        let d = e + c;
        let f = Value::new_param("f", -2.0);
        let mut loss = d.clone() * f.clone();
        assert_eq!(loss.as_f64(), -8.0);
        loss.backward();
        assert_eq!(a.grad(), 6.0);
        assert_eq!(b.grad(), -4.0);
        assert_eq!(d.grad(), -2.0);
        assert_eq!(f.grad(), 4.0);
        assert_eq!(loss.grad(), 1.0);
    }
}
