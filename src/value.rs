use std::{cell::RefCell, fmt::Display, ops::Add, rc::Rc};

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
}

impl Value {
    pub fn new_param<T: AsRef<str>>(name: T, value: f64) -> Value {
        InnerValue::new(ValueType::Float(Some(name.as_ref().to_owned())), value).into()
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

#[derive(Debug)]
enum ValueType {
    Float(Option<String>),
    Sum(Value, Value),
}

#[derive(Debug)]
struct InnerValue {
    _type: ValueType,
    value: f64,
}

impl InnerValue {
    fn new(_type: ValueType, value: f64) -> Self {
        InnerValue { _type, value }
    }
}

impl Display for InnerValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self._type {
            ValueType::Float(Some(name)) => write!(f, "{name}"),
            ValueType::Float(None) => write!(f, "{}", self.value),
            ValueType::Sum(a, b) => write!(f, "{} + {}", a, b),
        }
    }
}
