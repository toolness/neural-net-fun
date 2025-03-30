use std::{cell::RefCell, fmt::Display, ops::Add};

#[derive(Debug, Clone, Copy)]
pub struct NodeId(usize);

impl NodeId {
    pub fn compute(&self) -> f64 {
        CONTEXT.with_borrow(|ctx| ctx.compute(*self))
    }

    pub fn set(&mut self, value: f64) {
        CONTEXT.with_borrow_mut(|ctx| ctx.set(*self, value))
    }
}

impl Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", CONTEXT.with_borrow(|ctx| ctx.expr(*self)))
    }
}

impl Add<NodeId> for NodeId {
    type Output = NodeId;

    fn add(self, rhs: NodeId) -> Self::Output {
        CONTEXT.with_borrow_mut(|ctx| ctx.sum(self, rhs))
    }
}

impl From<f64> for NodeId {
    fn from(value: f64) -> Self {
        CONTEXT.with_borrow_mut(|ctx| ctx.param(String::default(), value))
    }
}

pub fn new_param<T: AsRef<str>>(name: T, value: f64) -> NodeId {
    CONTEXT.with_borrow_mut(|ctx| ctx.param(name.as_ref().to_owned(), value))
}

enum Node {
    Float(String, f64),
    Sum(NodeId, NodeId),
}

thread_local! {
    static CONTEXT: RefCell<Context> = RefCell::new(Context::default());
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
        let Node::Float(_, value) = self.get_node_mut(node) else {
            panic!("node must be a float");
        };
        *value = new_value;
    }

    pub fn param(&mut self, name: String, value: f64) -> NodeId {
        self.new_node(Node::Float(name, value))
    }

    pub fn sum(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.new_node(Node::Sum(a, b))
    }

    pub fn compute(&self, node: NodeId) -> f64 {
        match self.get_node(node) {
            Node::Float(_, node) => *node,
            Node::Sum(a, b) => self.compute(*a) + self.compute(*b),
        }
    }

    pub fn expr(&self, node: NodeId) -> String {
        match self.get_node(node) {
            Node::Float(name, value) => {
                if name.len() > 0 {
                    name.clone()
                } else {
                    format!("{value}")
                }
            }
            Node::Sum(a, b) => format!("({} + {})", self.expr(*a), self.expr(*b)),
        }
    }
}
