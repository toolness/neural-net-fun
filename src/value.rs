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

    fn order_topologically(&self) -> Vec<Value> {
        let visited: HashSet<i64> = HashSet::new();
        let mut result = vec![];
        let mut to_visit = vec![self.clone()];

        while let Some(value) = to_visit.pop() {
            let ptr = Rc::as_ptr(&value.0) as i64;
            if visited.contains(&ptr) {
                continue;
            }
            result.push(value.clone());
            to_visit.extend(value.children());
        }

        result
    }

    pub fn backward(&mut self) {
        self.0.borrow_mut().grad = 1.0;
        let values = self.order_topologically();
        for mut value in values {
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
