use std::{cell::RefCell, ops::Add, rc::Rc};

#[derive(Debug, Clone, Copy)]
pub struct NodeId(usize);

pub enum Node {
    Float(f64),
    Sum(NodeId, NodeId),
}

#[derive(Default)]
pub struct Context {
    nodes: Vec<Node>,
}

impl Context {
    fn new_node(&mut self, node: Node) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(node);
        NodeId(id)
    }

    fn get_node(&self, node: NodeId) -> &Node {
        self.nodes.get(node.0).expect("node id must exist")
    }

    fn get_node_mut(&mut self, node: NodeId) -> &mut Node {
        self.nodes.get_mut(node.0).expect("node id must exist")
    }

    pub fn set(&mut self, node: NodeId, new_value: f64) {
        let Node::Float(value) = self.get_node_mut(node) else {
            panic!("node must be a float");
        };
        *value = new_value;
    }

    pub fn param(&mut self, value: f64) -> NodeId {
        self.new_node(Node::Float(value))
    }

    pub fn sum(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.new_node(Node::Sum(a, b))
    }

    pub fn compute(&self, node: NodeId) -> f64 {
        match self.get_node(node) {
            Node::Float(node) => *node,
            Node::Sum(a, b) => self.compute(*a) + self.compute(*b),
        }
    }
}

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

    {
        let mut ctx = Context::default();
        let x = ctx.param(1.0);
        let y = ctx.param(3.0);
        let sum = ctx.sum(x, y);
        println!("{}", ctx.compute(sum));
        ctx.set(x, 2.0);
        println!("{}", ctx.compute(sum));
    }
}
