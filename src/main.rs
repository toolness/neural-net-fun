pub trait Value {
    fn data(&self) -> f64;
}

pub struct FloatValue {
    data: f64
}

impl FloatValue {
    fn new(data: f64) -> Self {
        FloatValue { data }
    }
}

impl Value for FloatValue {
    fn data(&self) -> f64 {
        self.data
    }
}

pub struct SumValue {
    left: Box<dyn Value>,
    right: Box<dyn Value>
}

impl Value for SumValue {
    fn data(&self) -> f64 {
        self.left.data() + self.right.data()
    }
}

fn main() {
    println!("hi");
}
